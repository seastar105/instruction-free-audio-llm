from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any


def preflight_vllm(export_dir: str | Path) -> dict[str, Any]:
    if importlib.metadata.version("vllm") != "0.27.1":
        raise RuntimeError("vLLM 0.27.1 is required")
    import torch
    import transformers
    from transformers import AutoConfig, AutoTokenizer
    from vllm import ModelRegistry

    from audio_lfm.vllm_plugin import ARCHITECTURE, register
    from audio_lfm.vllm_plugin.config import validate_export_config

    register()
    register()
    if ARCHITECTURE not in ModelRegistry.get_supported_archs():
        raise RuntimeError("AudioLFM2 plugin discovery failed")
    config = AutoConfig.from_pretrained(export_dir, trust_remote_code=False)
    validate_export_config(config)
    tokenizer = AutoTokenizer.from_pretrained(export_dir, trust_remote_code=False)
    if tokenizer.encode(config.audio_token, add_special_tokens=False) != [
        config.audio_token_index
    ]:
        raise RuntimeError("Exported audio placeholder contract failed")
    return {
        "vllm": importlib.metadata.version("vllm"),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "architecture": ARCHITECTURE,
        "config_format": config.audio_lfm_format_version,
    }


def evaluate_from_config(
    config_path: str | Path, *, split: str, allow_final_evaluation: bool
) -> None:
    import torch
    import yaml
    from transformers import AutoConfig, AutoTokenizer, WhisperFeatureExtractor
    from vllm import LLM, SamplingParams

    from audio_lfm.data.captionstew_backend import CaptionStewBackend
    from audio_lfm.data.catalog import CatalogIndex
    from audio_lfm.evaluation.predictions import PredictionWriter
    from audio_lfm.model.frontends.whisper_math import whisper_encoder_lengths
    from audio_lfm.prompts import build_vllm_audio_request_prompt
    from audio_lfm.utils.hashing import canonical_sha256, sha256_text
    from audio_lfm.vllm_plugin.types import make_audio_uuid

    if split == "test":
        raise ValueError("ParaSpeechCaps test evaluation is forbidden")
    if split == "holdout" and not allow_final_evaluation:
        raise ValueError("Holdout requires --allow-final-evaluation")
    if split not in {"dev", "holdout"}:
        raise ValueError("vLLM evaluation split must be dev or holdout")
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    export_dir = Path(config["model"]["export_dir"])
    preflight_vllm(export_dir)
    export_config = AutoConfig.from_pretrained(export_dir, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(export_dir, trust_remote_code=False)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        export_config.audio_model_id,
        revision=export_config.audio_model_revision,
    )
    v = config["vllm"]
    llm = LLM(
        model=str(export_dir),
        dtype=v["dtype"],
        tensor_parallel_size=v["tensor_parallel_size"],
        pipeline_parallel_size=v["pipeline_parallel_size"],
        max_model_len=v["max_model_len"],
        gpu_memory_utilization=v["gpu_memory_utilization"],
        max_num_seqs=v["max_num_seqs"],
        max_num_batched_tokens=v["max_num_batched_tokens"],
        limit_mm_per_prompt=v["limit_mm_per_prompt"],
        enforce_eager=v["enforce_eager"],
        enable_prefix_caching=v["enable_prefix_caching"],
        enable_mm_embeds=v["enable_mm_embeds"],
        mm_processor_cache_gb=v["mm_processor_cache_gb"],
        mm_processor_cache_type=v["mm_processor_cache_type"],
        trust_remote_code=False,
        load_format="safetensors",
    )
    generation = config["generation"]
    sampling_params = SamplingParams(
        temperature=generation["temperature"],
        top_p=generation["top_p"],
        top_k=generation["top_k"],
        repetition_penalty=generation["repetition_penalty"],
        max_tokens=generation["max_tokens"],
        seed=generation["seed"],
    )
    data = config["data"]
    root = str(data["captionstew_root"])
    if root.startswith("${ENV:") and root.endswith("}"):
        variable = root[6:-1]
        if variable not in os.environ:
            raise ValueError(f"Required environment variable {variable!r} is not set")
        root = os.environ[variable]
    catalog = CatalogIndex.load(
        root=root,
        dataset=data["dataset"],
        logical_split=split,
    )
    backend = CaptionStewBackend(
        captionstew_root=root,
        dataset=data["dataset"],
        catalog=catalog,
        shard_shuffle=0,
        sample_shuffle=0,
        max_audio_seconds=float(data["max_audio_seconds"]),
        long_audio_policy=data["long_audio_policy"],
        strict_target_consistency=True,
        max_bad_samples=0,
        seed=int(config["run"]["seed"]),
    )
    output_dir = Path(config["run"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = PredictionWriter(
        output_dir, part_size=int(config["run"]["prediction_part_size"])
    )
    completed = writer.completed_audio_ids()
    rendered, _ = build_vllm_audio_request_prompt(
        tokenizer=tokenizer, audio_token=export_config.audio_token
    )
    if rendered.count(export_config.audio_token) != 1:
        raise RuntimeError("vLLM prompt must contain exactly one audio token")
    frontend_hash = canonical_sha256(
        {
            "audio_model_revision": export_config.audio_model_revision,
            "audio_config": export_config.audio_config,
            "frontend_mode": export_config.frontend_mode,
        }
    )
    evaluation_manifest = {
        "format_version": 1,
        "split": split,
        "catalog_fingerprint": catalog.fingerprint,
        "projector_checkpoint_sha256": export_config.projector_checkpoint_sha256,
        "text_model_revision": export_config.text_model_revision,
        "audio_model_revision": export_config.audio_model_revision,
        "prompt_sha256": export_config.prompt_sha256,
        "chat_template_sha256": export_config.base_chat_template_sha256,
        "generation": generation,
        "vllm_version": importlib.metadata.version("vllm"),
    }
    evaluation_hash = canonical_sha256(evaluation_manifest)
    manifest_path = output_dir / "evaluation_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if canonical_sha256(existing) != evaluation_hash:
            raise RuntimeError("Incompatible vLLM evaluation resume manifest")
    else:
        manifest_path.write_text(
            json.dumps(evaluation_manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    chunk_items: list[tuple[Any, dict[str, Any]]] = []
    chunk_seconds = 0.0
    max_items = int(data["request_chunk_max_items"])
    max_seconds = float(data["request_chunk_max_audio_seconds"])
    evaluated = 0
    evaluation_limit = config["evaluation"].get("max_audio_items")

    def flush_chunk() -> None:
        nonlocal chunk_items, chunk_seconds
        if not chunk_items:
            return
        requests = [request for _, request in chunk_items]
        outputs = llm.generate(
            requests, sampling_params=sampling_params, use_tqdm=False
        )
        if len(outputs) != len(chunk_items):
            raise RuntimeError("vLLM output-count mismatch")
        for (raw, _), output in zip(chunk_items, outputs, strict=True):
            completion = output.outputs[0]
            catalog_record = catalog.audio_by_id[raw.audio_id]
            writer.add(
                {
                    "audio_id": raw.audio_id,
                    "source_id": raw.source_id,
                    "dataset": data["dataset"],
                    "split": split,
                    "generated_text": completion.text,
                    "generated_token_ids": list(completion.token_ids),
                    "finish_reason": completion.finish_reason,
                    "truncated": completion.finish_reason == "length",
                    "stop_reason": str(completion.stop_reason),
                    "cumulative_logprob": completion.cumulative_logprob,
                    "reference_target_ids": [
                        target.target_id for target in raw.style_captions
                    ],
                    "reference_texts": (
                        [target.text for target in raw.style_captions]
                        if config["evaluation"]["include_reference_text_in_predictions"]
                        else []
                    ),
                    "reference_count": len(raw.style_captions),
                    "audio_duration_seconds": raw.waveform.numel() / 16_000,
                    "original_num_samples": raw.original_num_samples,
                    "crop_start_sample": raw.crop_start_sample,
                    "evaluated_num_samples": raw.waveform.numel(),
                    "flac_sha256": catalog_record.flac_sha256,
                    "input_prompt_sha256": sha256_text(rendered),
                    "prompt_template_sha256": export_config.prompt_sha256,
                    "chat_template_sha256": export_config.base_chat_template_sha256,
                    "projector_checkpoint_sha256": (
                        export_config.projector_checkpoint_sha256
                    ),
                    "training_run_manifest_sha256": (
                        export_config.training_run_manifest_sha256
                    ),
                    "text_model_id": export_config.text_model_id,
                    "text_model_revision": export_config.text_model_revision,
                    "audio_model_id": export_config.audio_model_id,
                    "audio_model_revision": export_config.audio_model_revision,
                    "vllm_version": importlib.metadata.version("vllm"),
                    "export_format_version": export_config.audio_lfm_format_version,
                    "sampling_temperature": generation["temperature"],
                    "sampling_top_p": generation["top_p"],
                    "sampling_top_k": generation["top_k"],
                    "sampling_repetition_penalty": generation["repetition_penalty"],
                    "sampling_max_tokens": generation["max_tokens"],
                    "sampling_seed": generation["seed"],
                    "provenance": json.dumps(raw.metadata, sort_keys=True),
                    "evaluation_manifest_sha256": evaluation_hash,
                }
            )
        writer.flush()
        chunk_items = []
        chunk_seconds = 0.0

    for raw in backend.iter_epoch(0):
        if raw.audio_id in completed:
            continue
        if evaluation_limit is not None and evaluated >= int(evaluation_limit):
            break
        feature_batch = feature_extractor(
            [raw.waveform.numpy()],
            sampling_rate=16_000,
            padding="longest",
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        features = feature_batch.input_features[0].float().contiguous()
        feature_length = int(feature_batch.attention_mask[0].sum().item())
        encoder_length = int(
            whisper_encoder_lengths(torch.tensor([feature_length]))[0].item()
        )
        stack_factor = int(export_config.projector_config["stack_factor"])
        token_length = (encoder_length + stack_factor - 1) // stack_factor + 2
        audio_uuid = make_audio_uuid(
            audio_id=raw.audio_id,
            flac_sha256=catalog.audio_by_id[raw.audio_id].flac_sha256,
            crop_start_sample=raw.crop_start_sample,
            num_samples=raw.waveform.numel(),
            frontend_config_sha256=frontend_hash,
            projector_checkpoint_sha256=(export_config.projector_checkpoint_sha256),
        )
        request = {
            "prompt": rendered,
            "multi_modal_data": {
                "audio": {
                    "audio_features": features,
                    "audio_feature_length": torch.tensor(feature_length),
                    "audio_token_length": torch.tensor(token_length),
                }
            },
            "multi_modal_uuids": {"audio": audio_uuid},
        }
        seconds = raw.waveform.numel() / 16_000
        if chunk_items and (
            len(chunk_items) >= max_items or chunk_seconds + seconds > max_seconds
        ):
            flush_chunk()
        chunk_items.append((raw, request))
        chunk_seconds += seconds
        evaluated += 1
    flush_chunk()
    (output_dir / "progress.json").write_text(
        json.dumps(
            {"completed_audio_items": len(writer.completed_audio_ids())}, indent=2
        ),
        encoding="utf-8",
    )
