from __future__ import annotations

from pathlib import Path
from typing import Any


def benchmark_from_config(config_path: str | Path) -> dict[str, Any]:
    import yaml

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return {
        "export_dir": config["model"]["export_dir"],
        "max_num_seqs": config["vllm"]["max_num_seqs"],
        "max_num_batched_tokens": config["vllm"]["max_num_batched_tokens"],
        "audio_encoder_microbatch_size": config["vllm"][
            "audio_encoder_microbatch_size"
        ],
        "enforce_eager": config["vllm"]["enforce_eager"],
    }
