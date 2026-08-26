from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from typing import Any

import torch


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_environment() -> dict[str, Any]:
    driver = None
    if torch.cuda.is_available():
        try:
            driver = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()[0]
        except (OSError, subprocess.SubprocessError, IndexError):
            driver = None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_driver": driver,
        "cuda_available": torch.cuda.is_available(),
        "torchaudio": _version("torchaudio"),
        "transformers": _version("transformers"),
        "flash_attn": _version("flash-attn"),
        "causal_conv1d": _version("causal-conv1d"),
        "webdataset": _version("webdataset"),
        "pyarrow": _version("pyarrow"),
        "captionstew": _version("captionstew"),
        "audio_lfm": _version("audio-lfm"),
    }


def require_cuda_environment() -> dict[str, Any]:
    versions = collect_environment()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 support is required")
    missing = [
        name for name in ("flash_attn", "causal_conv1d") if versions[name] is None
    ]
    if missing:
        raise RuntimeError(f"Missing CUDA extensions: {', '.join(missing)}")
    return versions
