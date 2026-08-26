from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from typing import Annotated

import torch
import typer
from rich.console import Console

from audio_lfm.config import AppConfig, load_config
from audio_lfm.data.captionstew_backend import CaptionStewBackend
from audio_lfm.data.catalog import CatalogIndex
from audio_lfm.data.decode import DurationSidecar
from audio_lfm.data.loader import build_epoch_dataloader
from audio_lfm.data.local_shards import complete_local_shards
from audio_lfm.data.mixed_backend import MixedCaptionStewBackend
from audio_lfm.data.packed_loader import (
    WorkerPackingConfig,
    build_packed_epoch_dataloader,
)
from audio_lfm.data.worker_preprocessing import AudioPreprocessingConfig
from audio_lfm.environment import collect_environment, require_cuda_environment

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
console = Console()
ConfigOption = Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)]


def _catalog(config: AppConfig, split: str) -> CatalogIndex:
    return CatalogIndex.load(
        root=config.data.captionstew_root,
        dataset=config.data.dataset,
        logical_split=split,
        target_type=config.data.target_type,
        target_provider=config.data.target_provider,
        review_status_allowlist=(
            set(config.data.review_status_allowlist)
            if config.data.review_status_allowlist is not None
            else None
        ),
    )


def _backend(
    config: AppConfig,
    catalog: CatalogIndex,
    *,
    dataset: str | None = None,
    duration_sidecar: DurationSidecar | None = None,
) -> CaptionStewBackend:
    return CaptionStewBackend(
        captionstew_root=config.data.captionstew_root,
        dataset=dataset or config.data.dataset,
        catalog=catalog,
        shard_shuffle=config.data.shard_shuffle,
        sample_shuffle=config.data.sample_shuffle,
        max_audio_seconds=config.data.max_audio_seconds,
        long_audio_policy=config.data.long_audio_policy,
        strict_target_consistency=config.data.strict_target_consistency,
        max_bad_samples=config.data.max_bad_samples,
        seed=config.run.seed,
        duration_sidecar=duration_sidecar,
    )


def _training_catalogs(config: AppConfig) -> tuple[CatalogIndex, ...]:
    sources = config.data.training_sources
    if sources is None:
        return (_catalog(config, config.data.train_split),)
    return tuple(
        CatalogIndex.load(
            root=config.data.captionstew_root,
            dataset=source.dataset,
            logical_split=source.splits,
            target_type=config.data.target_type,
            target_provider=config.data.target_provider,
            review_status_allowlist=(
                set(config.data.review_status_allowlist)
                if config.data.review_status_allowlist is not None
                else None
            ),
        )
        for source in sources
    )


def _training_backend(
    config: AppConfig, catalogs: tuple[CatalogIndex, ...]
) -> MixedCaptionStewBackend:
    duration_sidecar = (
        DurationSidecar(
            config.data.duration_sidecar,
            require_exact=config.data.require_exact_duration_sidecar,
        )
        if config.data.duration_sidecar is not None
        else None
    )
    if config.data.require_exact_duration_sidecar:
        if duration_sidecar is None:
            raise ValueError("Exact duration metadata is required for training")
        for catalog in catalogs:
            missing = [
                record.audio_id
                for record in catalog.audio_by_id.values()
                if duration_sidecar.get(record) is None
            ]
            if missing:
                raise ValueError(
                    f"Exact duration sidecar lacks {len(missing)} rows for "
                    f"{catalog.dataset}; first={missing[0]!r}"
                )
    if config.data.require_complete_local_shards:
        for catalog in catalogs:
            if complete_local_shards(config.data.captionstew_root, catalog) is None:
                raise ValueError(
                    f"Complete local shards are required for {catalog.dataset}"
                )
    backends = tuple(
        _backend(
            config,
            catalog,
            dataset=catalog.dataset,
            duration_sidecar=duration_sidecar,
        )
        for catalog in catalogs
    )
    return MixedCaptionStewBackend(backends, seed=config.run.seed)


def _preprocessing_config(config: AppConfig) -> AudioPreprocessingConfig:
    if config.frontend.kind != "whisper":
        raise ValueError("Worker preprocessing currently requires Whisper")
    return AudioPreprocessingConfig(
        model_id=config.frontend.model_id,
        revision=config.frontend.revision,
        sample_rate=config.frontend.sample_rate,
        block_seconds=config.frontend.max_seconds,
    )


@app.command("preflight")
def preflight(
    config: ConfigOption,
    check_private_data: Annotated[bool, typer.Option("--check-private-data")] = False,
) -> None:
    """Check CUDA, pinned kernels, disk, and optionally private data."""
    loaded = load_config(config)
    versions = require_cuda_environment()
    from audio_lfm.model.loading import load_audio_lfm
    from audio_lfm.model.packing_preflight import (
        run_direct_causal_conv_boundary_test,
        run_lfm_packing_isolation_test,
    )

    result = run_direct_causal_conv_boundary_test()
    model, model_metadata = load_audio_lfm(loaded)
    lfm_result = run_lfm_packing_isolation_test(model.llm, device=torch.device("cuda"))
    disk_path = loaded.run.output_dir.parent
    while not disk_path.exists() and disk_path != disk_path.parent:
        disk_path = disk_path.parent
    free = shutil.disk_usage(disk_path).free
    if free < 5 * 1024**3:
        raise typer.BadParameter("Less than 5 GiB free for checkpoints")
    if check_private_data:
        from huggingface_hub import HfApi

        try:
            HfApi().whoami()
        except Exception as error:
            raise typer.BadParameter(
                "Hugging Face authentication is required; run `hf auth login`"
            ) from error
        catalogs = _training_catalogs(loaded)
        for catalog in catalogs:
            private_backend = _backend(loaded, catalog, dataset=catalog.dataset)
            private_backend.strict_target_consistency = (
                loaded.data.target_provider == "official_target"
            )
            checked = sum(1 for _ in islice(private_backend.iter_epoch(0), 100))
            if checked == 0:
                raise RuntimeError(
                    f"Private stream returned no samples for {catalog.dataset}"
                )
            console.print(
                f"Private stream OK for {catalog.dataset}: "
                f"decoded and cross-checked {checked} samples"
            )
    console.print_json(
        data={
            "environment": versions,
            "model": model_metadata,
            "causal_conv_boundary": asdict(result),
            "lfm_boundary": asdict(lfm_result),
        }
    )


@app.command("inspect-data")
def inspect_data(
    config: ConfigOption,
    num_samples: Annotated[int, typer.Option("--num-samples", min=1)] = 256,
) -> None:
    """Stream bounded data and report contract statistics without retaining audio."""
    loaded = load_config(config)
    catalogs = _training_catalogs(loaded)
    backend = _training_backend(loaded, catalogs)
    backend.strict_target_consistency = loaded.data.target_provider == "official_target"
    durations: list[float] = []
    transcripts = 0
    seen: set[str] = set()
    for index, example in enumerate(backend.iter_epoch(0)):
        if example.audio_id in seen:
            raise RuntimeError(f"Duplicate streamed audio_id: {example.audio_id}")
        seen.add(example.audio_id)
        durations.append(example.waveform.numel() / 16_000)
        transcripts += example.transcript is not None
        if index + 1 >= num_samples:
            break
    durations.sort()
    quantiles = (
        {
            str(q): durations[min(len(durations) - 1, int(q * (len(durations) - 1)))]
            for q in (0.0, 0.5, 0.9, 0.99, 1.0)
        }
        if durations
        else {}
    )
    console.print_json(
        data={
            "streamed": len(seen),
            "allowed": sum(len(catalog.allowed_audio_ids) for catalog in catalogs),
            "selected_shards": sum(
                len(catalog.selected_shards) for catalog in catalogs
            ),
            "duration_quantiles": quantiles,
            "transcript_presence": transcripts,
            "reference_count_distribution": {
                catalog.dataset: catalog.target_count_distribution
                for catalog in catalogs
            },
            "review_status_distribution": {
                catalog.dataset: catalog.review_status_distribution
                for catalog in catalogs
            },
            "split_overlap_report": {
                catalog.dataset: catalog.split_overlap_report for catalog in catalogs
            },
            "long_audio_skipped": sum(
                item.long_audio_skip_count for item in backend.backends
            ),
            "long_audio_exclusion_rate": (
                sum(item.long_audio_skip_count for item in backend.backends)
                / max(
                    1,
                    len(seen)
                    + sum(item.long_audio_skip_count for item in backend.backends),
                )
            ),
            "decode_failures": sum(
                item.decode_failure_count for item in backend.backends
            ),
        }
    )


@app.command("inspect-model")
def inspect_model(config: ConfigOption) -> None:
    """Load frozen models and report dimensions and trainable parameters."""
    loaded = load_config(config)
    from audio_lfm.model.loading import load_audio_lfm
    from audio_lfm.utils.tensors import count_parameters, tensor_rms

    model, metadata = load_audio_lfm(loaded)
    rates = {}
    for seconds in (1, 5, 10, 20, 30):
        frontend_length = int(
            model.frontend.estimate_output_lengths(torch.tensor([seconds * 16_000]))[0]
        )
        rates[str(seconds)] = model.projector.projected_length(frontend_length) + 2
    console.print_json(
        data={
            **metadata,
            "frontend_parameters": count_parameters(model.frontend),
            "projector_parameters": count_parameters(model.projector),
            "llm_parameters": count_parameters(model.llm),
            "embedding_rms": float(tensor_rms(model.llm.get_input_embeddings().weight)),
            "audio_token_lengths": rates,
            "trainable_names": [
                name for name, value in model.named_parameters() if value.requires_grad
            ],
        }
    )


@app.command("test-packing")
def test_packing(config: ConfigOption) -> None:
    """Run direct convolution and full LFM forward/backward isolation tests."""
    loaded = load_config(config)
    require_cuda_environment()
    from audio_lfm.model.loading import load_audio_lfm
    from audio_lfm.model.packing_preflight import (
        run_direct_causal_conv_boundary_test,
        run_lfm_packing_isolation_test,
    )

    direct = run_direct_causal_conv_boundary_test()
    model, _ = load_audio_lfm(loaded)
    lfm = run_lfm_packing_isolation_test(model.llm, device=torch.device("cuda"))
    console.print_json(data={"direct": asdict(direct), "lfm": asdict(lfm)})


@app.command("train")
def train(
    config: ConfigOption,
    resume: Annotated[Path | None, typer.Option("--resume", exists=True)] = None,
    allow_nonreproducible_resume: Annotated[
        bool, typer.Option("--allow-nonreproducible-resume")
    ] = False,
) -> None:
    """Train only the audio projector with a direct PyTorch loop."""
    loaded = load_config(config)
    if (
        loaded.data.duration_sidecar is not None
        and not loaded.data.duration_sidecar.is_file()
    ):
        raise ValueError(
            f"Duration sidecar does not exist: {loaded.data.duration_sidecar}"
        )
    catalogs = _training_catalogs(loaded)
    backend = _training_backend(loaded, catalogs)
    require_cuda_environment()
    if loaded.packing.require_boundary_kernel_tests:
        from audio_lfm.model.packing_preflight import (
            run_direct_causal_conv_boundary_test,
        )

        run_direct_causal_conv_boundary_test()
    from audio_lfm.model.loading import load_audio_lfm
    from audio_lfm.training.checkpoint import load_checkpoint
    from audio_lfm.training.engine import TrainingEngine
    from audio_lfm.training.optimizer import create_optimizer
    from audio_lfm.training.scheduler import cosine_warmup_scheduler
    from audio_lfm.utils.hashing import canonical_sha256
    from audio_lfm.utils.rng import seed_everything

    seed_everything(
        loaded.run.seed,
        deterministic_algorithms=loaded.run.deterministic_algorithms,
    )
    model, model_metadata = load_audio_lfm(loaded)
    expansion_provenance: dict[str, object] = {}
    if loaded.prompt.mode == "caption_expansion_alignment":
        from audio_lfm.overlays.decoder_lock import validate_decoder_lock

        expansion_provenance = validate_decoder_lock(loaded, model_metadata)
        if loaded.data.training_sources is not None:
            expansion_provenance["training_sources"] = [
                {
                    "dataset": source.dataset,
                    "splits": source.splits,
                    **validate_decoder_lock(
                        loaded,
                        model_metadata,
                        lock_path=source.expansion_decoder_lock,
                        expansion_recipe_sha256=(source.expansion_recipe_sha256),
                    ),
                }
                for source in loaded.data.training_sources
            ]
    if loaded.packing.require_boundary_kernel_tests:
        from audio_lfm.model.packing_preflight import run_lfm_packing_isolation_test

        run_lfm_packing_isolation_test(model.llm, device=torch.device("cuda"))
    compile_metadata: dict[str, object] = {"enabled": False}
    if loaded.optimization.torch_compile:
        compile_metadata = model.enable_torch_compile(
            backend=loaded.optimization.torch_compile_backend,
            mode=loaded.optimization.torch_compile_mode,
            dynamic=loaded.optimization.torch_compile_dynamic,
            compile_whisper_encoder=(loaded.optimization.compile_whisper_encoder),
            compile_projector=loaded.optimization.compile_projector,
            compile_lfm_backbone=loaded.optimization.compile_lfm_backbone,
        )
    optimizer = create_optimizer(
        model,
        learning_rate=loaded.optimization.learning_rate,
        weight_decay=loaded.optimization.weight_decay,
        betas=(loaded.optimization.beta1, loaded.optimization.beta2),
        eps=loaded.optimization.epsilon,
        fused=loaded.optimization.fused,
    )
    scheduler = cosine_warmup_scheduler(
        optimizer,
        warmup_updates=loaded.optimization.warmup_updates,
        max_updates=loaded.optimization.max_updates,
        min_learning_rate=loaded.optimization.min_learning_rate,
    )
    manifest = {
        **model_metadata,
        **expansion_provenance,
        "catalog_fingerprint": canonical_sha256(
            {
                catalog.dataset: {
                    "logical_split": catalog.logical_split,
                    "fingerprint": catalog.fingerprint,
                }
                for catalog in catalogs
            }
        ),
        "projector_architecture_sha256": canonical_sha256(
            loaded.projector.model_dump(mode="json")
        ),
        "data_semantics_sha256": canonical_sha256(
            {
                "training_sources": (
                    [
                        source.model_dump(mode="json")
                        for source in loaded.data.training_sources
                    ]
                    if loaded.data.training_sources is not None
                    else [
                        {
                            "dataset": loaded.data.dataset,
                            "splits": [loaded.data.train_split],
                        }
                    ]
                ),
                "validation_split": loaded.data.validation_split,
                "final_split": loaded.data.final_split,
                "target_type": loaded.data.target_type,
                "target_provider": loaded.data.target_provider,
            }
        ),
        "packing_semantics_sha256": canonical_sha256(
            loaded.packing.model_dump(mode="json")
        ),
        "environment": collect_environment(),
        "torch_compile": compile_metadata,
        "resolved_config": loaded.redacted_dict(),
    }
    trainer_state = data_state = None
    if resume is not None:
        trainer_state, data_state = load_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            current_manifest=manifest,
            allow_nonreproducible=allow_nonreproducible_resume,
        )
    engine = TrainingEngine(
        config=loaded,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        run_manifest=manifest,
        trainer_state=trainer_state,
        data_state=data_state,
        evaluation_callback=_training_evaluation_callback(loaded, model),
    )
    if loaded.data.duration_sidecar is None:
        raise ValueError(
            "Packed worker training requires data.duration_sidecar with num_samples"
        )
    worker_packing = WorkerPackingConfig(
        max_lfm_tokens=loaded.packing.max_lfm_tokens,
        max_sample_lfm_tokens=loaded.packing.sample_lfm_token_limit,
        planning_buffer_examples=loaded.packing.planning_buffer_examples,
        max_examples_per_pack=loaded.packing.max_examples_per_pack,
        oversized_example_policy=loaded.packing.oversized_example_policy,
        best_fit_decreasing=loaded.packing.best_fit_decreasing,
        stack_factor=loaded.projector.stack_factor,
        vocabulary_size=int(model.llm.config.vocab_size),
    )
    final = engine.train(
        lambda epoch, committed_audio_ids: build_packed_epoch_dataloader(
            backend,
            epoch=epoch,
            committed_audio_ids=committed_audio_ids,
            num_workers=loaded.data.num_workers,
            persistent_workers=loaded.data.persistent_workers,
            prefetch_factor=loaded.data.prefetch_factor,
            preprocessing=_preprocessing_config(loaded),
            prompt_compiler=model.prompt_compiler,
            packing=worker_packing,
        )
    )
    console.print_json(data=asdict(final))


def _training_evaluation_callback(
    loaded: AppConfig, model: object
) -> Callable[[int], float]:
    from audio_lfm.evaluation.teacher_forced import evaluate_all_references

    validation_catalog = _catalog(loaded, loaded.data.validation_split)
    validation_backend = _backend(loaded, validation_catalog)
    validation_mixed = MixedCaptionStewBackend(
        (validation_backend,), seed=loaded.run.seed
    )

    def run(update: int) -> float:
        examples = build_epoch_dataloader(
            validation_mixed,
            epoch=0,
            num_workers=loaded.data.num_workers,
            persistent_workers=loaded.data.persistent_workers,
            prefetch_factor=loaded.data.prefetch_factor,
            preprocessing=_preprocessing_config(loaded),
        )
        limit = loaded.evaluation.validation_max_audio_items
        if limit is not None:
            examples = islice(examples, limit)
        metrics = evaluate_all_references(model, examples, device=torch.device("cuda"))
        destination = Path(loaded.run.output_dir) / f"validation-{update:08d}.json"
        destination.write_text(
            json.dumps(asdict(metrics), indent=2, sort_keys=True), encoding="utf-8"
        )
        return metrics.audio_weighted_mean_nll

    return run


@app.command("evaluate")
def evaluate(
    config: ConfigOption,
    checkpoint: Annotated[Path, typer.Option("--checkpoint", exists=True)],
    split: Annotated[str, typer.Option("--split")] = "dev",
) -> None:
    """Run authoritative all-reference teacher-forced NLL evaluation."""
    loaded = load_config(config)
    if split == "test":
        raise typer.BadParameter("ParaSpeechCaps test evaluation is forbidden")
    if split not in {"dev", "holdout"}:
        raise typer.BadParameter("Evaluation split must be dev or holdout")
    from safetensors.torch import load_file

    from audio_lfm.evaluation.teacher_forced import evaluate_all_references
    from audio_lfm.model.loading import load_audio_lfm

    model, _ = load_audio_lfm(loaded)
    incompatible = model.load_state_dict(
        load_file(checkpoint / "projector.safetensors"), strict=False
    )
    missing_projector = [
        key for key in incompatible.missing_keys if key.startswith("projector.")
    ]
    if missing_projector:
        raise RuntimeError(f"Checkpoint lacks projector keys: {missing_projector}")
    catalog = _catalog(loaded, split)
    evaluation_backend = MixedCaptionStewBackend(
        (_backend(loaded, catalog),), seed=loaded.run.seed
    )
    examples = build_epoch_dataloader(
        evaluation_backend,
        epoch=0,
        num_workers=loaded.data.num_workers,
        persistent_workers=loaded.data.persistent_workers,
        prefetch_factor=loaded.data.prefetch_factor,
        preprocessing=_preprocessing_config(loaded),
    )
    limit = loaded.evaluation.validation_max_audio_items
    if limit is not None:
        examples = islice(examples, limit)
    metrics = evaluate_all_references(model, examples, device=torch.device("cuda"))
    console.print_json(data=asdict(metrics))


@app.command("generate")
def generate(
    config: ConfigOption,
    checkpoint: Annotated[Path, typer.Option("--checkpoint", exists=True)],
    split: Annotated[str, typer.Option("--split")] = "dev",
) -> None:
    """Generate a bounded qualitative set through the HF reference path."""
    loaded = load_config(config)
    if split == "test":
        raise typer.BadParameter("ParaSpeechCaps test evaluation is forbidden")
    from safetensors.torch import load_file

    from audio_lfm.evaluation.generation import generate_one
    from audio_lfm.evaluation.predictions import PredictionWriter
    from audio_lfm.model.loading import load_audio_lfm

    model, metadata = load_audio_lfm(loaded)
    model.load_state_dict(load_file(checkpoint / "projector.safetensors"), strict=False)
    writer = PredictionWriter(Path(loaded.run.output_dir) / f"generation-{split}")
    generation_backend = MixedCaptionStewBackend(
        (_backend(loaded, _catalog(loaded, split)),), seed=loaded.run.seed
    )
    generation_examples = build_epoch_dataloader(
        generation_backend,
        epoch=0,
        num_workers=loaded.data.num_workers,
        persistent_workers=loaded.data.persistent_workers,
        prefetch_factor=loaded.data.prefetch_factor,
        preprocessing=_preprocessing_config(loaded),
    )
    for index, raw in enumerate(generation_examples):
        if index >= loaded.evaluation.generation_examples:
            break
        text, tokens = generate_one(
            model,
            raw,
            device=torch.device("cuda"),
            max_new_tokens=loaded.evaluation.generation_max_new_tokens,
        )
        writer.add(
            {
                "audio_id": raw.audio_id,
                "source_id": raw.source_id,
                "generated_text": text,
                "generated_token_ids": tokens,
                "target_ids": [target.target_id for target in raw.style_captions],
                "crop_start_sample": raw.crop_start_sample,
                "metadata": json.dumps(raw.metadata, sort_keys=True),
                **metadata,
            }
        )
    writer.flush()


@app.command("expand-responses")
def expand_responses(
    catalog_dir: Annotated[Path, typer.Option("--catalog-dir", exists=True)],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    dataset: Annotated[str, typer.Option("--dataset")],
    source_target_types: Annotated[
        str, typer.Option("--source-target-types")
    ] = "caption",
    model_path: Annotated[Path, typer.Option("--model-path", exists=True)] = Path("."),
    model_id: Annotated[str, typer.Option("--model-id")] = (
        "LiquidAI/LFM2.5-1.2B-Instruct"
    ),
    model_revision: Annotated[str, typer.Option("--model-revision")] = "",
    max_targets: Annotated[int | None, typer.Option("--max-targets", min=1)] = None,
    max_tokens: Annotated[int, typer.Option("--max-tokens", min=1)] = 1024,
    request_batch_size: Annotated[
        int, typer.Option("--request-batch-size", min=1)
    ] = 8192,
    max_num_seqs: Annotated[int, typer.Option("--max-num-seqs", min=1)] = 512,
    max_num_batched_tokens: Annotated[
        int, typer.Option("--max-num-batched-tokens", min=1)
    ] = 32768,
    max_model_len: Annotated[int, typer.Option("--max-model-len", min=2)] = 2048,
    combine_paraspeech_sources: Annotated[
        bool, typer.Option("--combine-paraspeech-sources")
    ] = False,
) -> None:
    """Expand caption surrogates with large-batch vLLM into an overlay."""
    from audio_lfm.overlays.response_expansion import expand_catalog_with_vllm

    target_types = tuple(
        value.strip() for value in source_target_types.split(",") if value.strip()
    )
    manifest = expand_catalog_with_vllm(
        catalog_dir=catalog_dir,
        output_dir=output_dir,
        dataset_name=dataset,
        source_target_types=target_types,
        model_path=model_path,
        model_id=model_id,
        model_revision=model_revision,
        max_targets=max_targets,
        max_tokens=max_tokens,
        request_batch_size=request_batch_size,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        combine_style_caption_and_transcription=combine_paraspeech_sources,
    )
    console.print_json(data=manifest)


@app.command("benchmark")
def benchmark(config: ConfigOption, steps: int = 10) -> None:
    """Run the smoke training configuration for a caller-controlled update count."""
    loaded = load_config(config)
    console.print_json(
        data={
            "steps": steps,
            "packing_limit": loaded.packing.max_lfm_tokens,
            "mode": loaded.frontend.mode,
            "note": "Use the smoke train command for end-to-end timed updates.",
        }
    )


@app.command("export-vllm")
def export_vllm(
    config: ConfigOption,
    checkpoint: Annotated[Path, typer.Option("--checkpoint", exists=True)],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
) -> None:
    """Export the small projector-only vLLM artifact."""
    from audio_lfm.vllm_plugin.export import export_vllm_artifact

    export_vllm_artifact(load_config(config), checkpoint, output_dir)


@app.command("preflight-vllm")
def preflight_vllm(export_dir: Annotated[Path, typer.Option(exists=True)]) -> None:
    from audio_lfm.vllm_plugin.runner import preflight_vllm as run

    console.print_json(data=run(export_dir))


@app.command("evaluate-vllm")
def evaluate_vllm(
    config: ConfigOption,
    split: Annotated[str, typer.Option("--split")] = "dev",
    allow_final_evaluation: Annotated[
        bool, typer.Option("--allow-final-evaluation")
    ] = False,
) -> None:
    from audio_lfm.vllm_plugin.runner import evaluate_from_config

    evaluate_from_config(
        config, split=split, allow_final_evaluation=allow_final_evaluation
    )


@app.command("compare-hf-vllm")
def compare_hf_vllm(export_dir: Annotated[Path, typer.Option(exists=True)]) -> None:
    from audio_lfm.vllm_plugin.parity import parity_summary

    console.print_json(data=parity_summary(export_dir))


@app.command("benchmark-vllm")
def benchmark_vllm(config: ConfigOption) -> None:
    from audio_lfm.vllm_plugin.benchmark import benchmark_from_config

    console.print_json(data=benchmark_from_config(config))


if __name__ == "__main__":
    app()
