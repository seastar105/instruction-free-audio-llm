from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

from audio_lfm.config import load_config
from audio_lfm.data.captionstew_backend import CaptionStewBackend
from audio_lfm.data.catalog import CatalogIndex
from audio_lfm.data.decode import DurationSidecar
from audio_lfm.data.mixed_backend import MixedCaptionStewBackend
from audio_lfm.data.packed_loader import (
    WorkerPackingConfig,
    build_packed_epoch_dataloader,
)
from audio_lfm.data.worker_preprocessing import AudioPreprocessingConfig
from audio_lfm.model.prompt_compiler import PromptCompiler


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--duration-sidecar", type=Path)
    parser.add_argument("--planning-buffer-examples", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--packs", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    config = load_config(args.config)
    source = next(
        source
        for source in (config.data.training_sources or ())
        if source.dataset == args.dataset
    )
    catalog = CatalogIndex.load(
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
    if config.data.duration_sidecar is None:
        raise ValueError("Training config has no duration sidecar")
    backend = CaptionStewBackend(
        captionstew_root=config.data.captionstew_root,
        dataset=catalog.dataset,
        catalog=catalog,
        shard_shuffle=config.data.shard_shuffle,
        sample_shuffle=config.data.sample_shuffle,
        max_audio_seconds=config.data.max_audio_seconds,
        long_audio_policy=config.data.long_audio_policy,
        strict_target_consistency=config.data.strict_target_consistency,
        max_bad_samples=config.data.max_bad_samples,
        seed=config.run.seed,
        duration_sidecar=DurationSidecar(
            args.duration_sidecar or config.data.duration_sidecar
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.llm.model_id,
        revision=config.llm.revision,
        trust_remote_code=False,
        local_files_only=True,
    )
    llm_config = AutoConfig.from_pretrained(
        config.llm.model_id,
        revision=config.llm.revision,
        trust_remote_code=False,
        local_files_only=True,
    )
    compiler = PromptCompiler(
        tokenizer,
        prompt_file=config.prompt.prompt_file,
        audio_sentinel=config.prompt.audio_sentinel,
        mode=config.prompt.mode,
        system_message=config.prompt.system_message,
        supervise_assistant_termination=(config.prompt.supervise_assistant_termination),
    )
    loader = build_packed_epoch_dataloader(
        MixedCaptionStewBackend((backend,), seed=config.run.seed),
        epoch=0,
        committed_audio_ids=frozenset(),
        num_workers=args.num_workers,
        persistent_workers=False,
        prefetch_factor=config.data.prefetch_factor,
        preprocessing=AudioPreprocessingConfig(
            model_id=config.frontend.model_id,
            revision=config.frontend.revision,
            sample_rate=config.frontend.sample_rate,
            block_seconds=config.frontend.max_seconds,
        ),
        prompt_compiler=compiler,
        packing=WorkerPackingConfig(
            max_lfm_tokens=config.packing.max_lfm_tokens,
            max_sample_lfm_tokens=config.packing.sample_lfm_token_limit,
            planning_buffer_examples=args.planning_buffer_examples,
            max_examples_per_pack=config.packing.max_examples_per_pack,
            oversized_example_policy=config.packing.oversized_example_policy,
            best_fit_decreasing=config.packing.best_fit_decreasing,
            stack_factor=config.projector.stack_factor,
            vocabulary_size=int(llm_config.vocab_size),
        ),
    )
    result = []
    iterator = iter(loader)
    try:
        for item in islice(iterator, args.packs):
            result.append(
                {
                    "audio_ids": item.batch.layout.audio_ids,
                    "input_tokens": item.batch.layout.input_token_count,
                    "logical_examples": len(item.batch.layout.audio_ids),
                    "audio_blocks": item.batch.input_features.shape[0],
                    "audio_seconds": item.batch.audio_seconds,
                    "oversized_examples_skipped": item.oversized_examples_skipped,
                }
            )
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
    if len(result) != args.packs:
        raise RuntimeError(f"Expected {args.packs} packs, received {len(result)}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
