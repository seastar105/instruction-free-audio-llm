from __future__ import annotations

from pathlib import Path
from typing import Any


def parity_summary(export_dir: str | Path) -> dict[str, Any]:
    from audio_lfm.vllm_plugin.runner import preflight_vllm

    result = preflight_vllm(export_dir)
    result["status"] = (
        "preflight passed; supply identical local examples for token parity"
    )
    return result
