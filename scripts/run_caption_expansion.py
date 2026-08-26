from __future__ import annotations

import argparse
import json
from pathlib import Path

from audio_lfm.overlays.response_expansion import expand_catalog_with_vllm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand official caption surrogates with vLLM."
    )
    parser.add_argument("--catalog-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-target-types", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-id", default="LiquidAI/LFM2.5-1.2B-Instruct")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--request-batch-size", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--combine-paraspeech-sources", action="store_true")
    arguments = parser.parse_args()
    manifest = expand_catalog_with_vllm(
        catalog_dir=arguments.catalog_dir,
        output_dir=arguments.output_dir,
        dataset_name=arguments.dataset,
        source_target_types=tuple(
            item.strip()
            for item in arguments.source_target_types.split(",")
            if item.strip()
        ),
        model_path=arguments.model_path,
        model_id=arguments.model_id,
        model_revision=arguments.model_revision,
        max_targets=arguments.max_targets,
        max_tokens=arguments.max_tokens,
        request_batch_size=arguments.request_batch_size,
        max_num_seqs=arguments.max_num_seqs,
        max_num_batched_tokens=arguments.max_num_batched_tokens,
        max_model_len=arguments.max_model_len,
        combine_style_caption_and_transcription=(arguments.combine_paraspeech_sources),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
