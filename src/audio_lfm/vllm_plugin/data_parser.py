from __future__ import annotations

from typing import Any


def validate_feature_dict(data: dict[str, Any]) -> None:
    import torch

    required = {"audio_features", "audio_feature_length", "audio_token_length"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Audio feature item lacks fields: {sorted(missing)}")
    features = torch.as_tensor(data["audio_features"])
    if features.ndim != 2 or features.shape[0] != 80:
        raise ValueError("audio_features must be [80, mel_frames]")
    if not torch.isfinite(features).all():
        raise ValueError("audio_features contain non-finite values")
    feature_length = int(torch.as_tensor(data["audio_feature_length"]).item())
    token_length = int(torch.as_tensor(data["audio_token_length"]).item())
    if feature_length <= 0 or feature_length > features.shape[1]:
        raise ValueError("Invalid audio_feature_length")
    if token_length <= 2:
        raise ValueError("Invalid audio_token_length")
    chunk_lengths = torch.as_tensor(
        data.get("audio_chunk_length", [feature_length])
    ).flatten()
    if chunk_lengths.numel() == 0 or bool(torch.any(chunk_lengths <= 0)):
        raise ValueError("audio_chunk_length must contain positive lengths")
    if int(chunk_lengths.sum().item()) != feature_length:
        raise ValueError("audio_chunk_length must sum to audio_feature_length")


def build_vllm_data_parser_class() -> type[Any]:
    from vllm.multimodal.inputs import MultiModalFieldConfig
    from vllm.multimodal.parse import DictEmbeddingItems, MultiModalDataParser

    fields = {
        "audio_features": MultiModalFieldConfig.batched("audio"),
        "audio_feature_length": MultiModalFieldConfig.batched("audio"),
        "audio_chunk_length": MultiModalFieldConfig.batched("audio"),
        "audio_token_length": MultiModalFieldConfig.batched("audio"),
        "audio_embeds": MultiModalFieldConfig.batched("audio"),
    }

    class AudioLfm2MultiModalDataParser(MultiModalDataParser):
        def _parse_audio_data(self, data: Any) -> Any:
            if isinstance(data, dict):
                validate_feature_dict(data)
                return DictEmbeddingItems(
                    data,
                    modality="audio",
                    required_fields={
                        "audio_features",
                        "audio_feature_length",
                        "audio_token_length",
                    },
                    fields_factory=lambda _: fields,
                )
            return super()._parse_audio_data(data)

    return AudioLfm2MultiModalDataParser
