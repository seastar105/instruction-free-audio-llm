from __future__ import annotations

from typing import Any

REQUIRED_EXPORT_FIELDS = {
    "audio_lfm_format_version",
    "text_model_id",
    "text_model_revision",
    "audio_model_id",
    "audio_model_revision",
    "audio_token",
    "audio_token_index",
    "text_tokenizer_size",
    "audio_config",
    "projector_config",
    "projector_checkpoint_sha256",
}


def validate_export_config(config: Any) -> None:
    missing = [field for field in REQUIRED_EXPORT_FIELDS if not hasattr(config, field)]
    if missing:
        raise ValueError(f"Export config lacks fields: {sorted(missing)}")
    if config.model_type != "lfm2":
        raise ValueError("Export config must retain model_type='lfm2'")
    if config.architectures != ["AudioLfm2ForConditionalGeneration"]:
        raise ValueError("Export config architecture mismatch")
    if config.text_tokenizer_size <= 0:
        raise ValueError("Frozen tokenizer size must be positive")
    if config.audio_token_index < config.text_tokenizer_size:
        raise ValueError("Audio token must be outside the frozen tokenizer vocabulary")
    for name in ("text_model_revision", "audio_model_revision"):
        value = str(getattr(config, name))
        if len(value) != 40 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"{name} must be an immutable 40-character commit SHA")


def projector_config_from_training(
    config: dict[str, Any], *, input_dim: int, output_dim: int, target_rms: float
) -> dict[str, Any]:
    return {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "stack_factor": int(config["stack_factor"]),
        "hidden_dim": int(config["hidden_dim"]),
        "dropout": float(config.get("dropout", 0.0)),
        "use_input_layer_norm": bool(config.get("use_input_layer_norm", True)),
        "use_output_rms_norm": bool(config.get("use_output_rms_norm", True)),
        "target_embedding_rms": target_rms,
    }
