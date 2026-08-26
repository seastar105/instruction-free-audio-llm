from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from audio_lfm.config import AppConfig
from audio_lfm.prompts import audio_training_prompt_template_sha256


class DecoderLockError(ValueError):
    """Caption expansion and audio training use incompatible decoders."""


def _require_equal(
    lock: Mapping[str, Any],
    model_metadata: Mapping[str, Any],
    lock_key: str,
    model_key: str,
) -> None:
    if lock.get(lock_key) != model_metadata.get(model_key):
        raise DecoderLockError(
            f"Decoder lock mismatch for {lock_key}: "
            f"{lock.get(lock_key)!r} != {model_metadata.get(model_key)!r}"
        )


def validate_decoder_lock(
    config: AppConfig,
    model_metadata: Mapping[str, Any],
    *,
    lock_path: Path | None = None,
    expansion_recipe_sha256: str | None = None,
) -> dict[str, object]:
    if config.prompt.mode != "caption_expansion_alignment":
        raise DecoderLockError("Decoder lock applies only to caption expansion mode")
    resolved_lock_path = lock_path or cast(Path, config.data.expansion_decoder_lock)
    lock = json.loads(resolved_lock_path.read_text(encoding="utf-8"))
    if lock.get("format_version") != 2:
        raise DecoderLockError("Caption expansion decoder lock format must be 2")
    _require_equal(lock, model_metadata, "model_id", "llm_model_id")
    _require_equal(lock, model_metadata, "model_revision", "llm_revision")
    _require_equal(lock, model_metadata, "tokenizer_revision", "tokenizer_revision")
    _require_equal(lock, model_metadata, "chat_template_sha256", "chat_template_sha256")
    expected_recipe = expansion_recipe_sha256 or config.data.expansion_recipe_sha256
    if lock.get("expansion_recipe_sha256") != expected_recipe:
        raise DecoderLockError(
            "Configured expansion recipe does not match decoder lock"
        )
    training_hash = audio_training_prompt_template_sha256(config.prompt.audio_sentinel)
    expansion_hash = str(lock.get("expansion_prompt_template_sha256", ""))
    if training_hash == expansion_hash:
        raise DecoderLockError(
            "Expansion and audio-training prompts unexpectedly match. The "
            "expansion system instruction must not be used during training."
        )
    return {
        "expansion_recipe_sha256": lock["expansion_recipe_sha256"],
        "expansion_system_prompt_sha256": lock["expansion_system_prompt_sha256"],
        "expansion_prompt_template_sha256": expansion_hash,
        "audio_training_has_system_message": False,
        "audio_training_prompt_template_sha256": training_hash,
        "decoder_model_id": lock["model_id"],
        "decoder_revision": lock["model_revision"],
        "tokenizer_revision": lock["tokenizer_revision"],
        "chat_template_sha256": lock["chat_template_sha256"],
    }
