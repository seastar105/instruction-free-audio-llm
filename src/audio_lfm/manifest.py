from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from audio_lfm.config import AppConfig
from audio_lfm.environment import collect_environment
from audio_lfm.prompts import audio_training_prompt_template_sha256
from audio_lfm.utils.hashing import sha256_text


@dataclass
class RunManifest:
    format_version: int = 1
    environment: dict[str, Any] = field(default_factory=collect_environment)
    llm_revision: str | None = None
    frontend_revision: str | None = None
    tokenizer_revision: str | None = None
    chat_template_sha256: str | None = None
    prompt_sha256: str | None = None
    catalog_fingerprint: str | None = None
    projector_embedding_target_rms: float | None = None
    resolved_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: AppConfig) -> RunManifest:
        if config.prompt.mode == "caption_expansion_alignment":
            prompt_hash = audio_training_prompt_template_sha256(
                config.prompt.audio_sentinel
            )
        else:
            if config.prompt.prompt_file is None:
                raise ValueError("Direct-caption alignment requires a prompt file")
            prompt_hash = sha256_text(
                config.prompt.prompt_file.read_text(encoding="utf-8")
            )
        return cls(
            prompt_sha256=prompt_hash,
            resolved_config=config.redacted_dict(),
        )

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8"
        )
