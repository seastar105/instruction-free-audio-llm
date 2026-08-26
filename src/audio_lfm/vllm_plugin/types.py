from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


def make_audio_uuid(
    *,
    audio_id: str,
    flac_sha256: str,
    crop_start_sample: int | None,
    num_samples: int,
    frontend_config_sha256: str,
    projector_checkpoint_sha256: str,
) -> str:
    payload = "\0".join(
        [
            audio_id,
            flac_sha256,
            str(crop_start_sample),
            str(num_samples),
            frontend_config_sha256,
            projector_checkpoint_sha256,
        ]
    ).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class AudioFeatureItem:
    audio_id: str
    audio_features: np.ndarray
    audio_feature_length: int
    audio_token_length: int
    audio_uuid: str
    rendered_prompt: str

    def validate(self) -> None:
        if self.audio_features.ndim != 2 or self.audio_features.shape[0] != 80:
            raise ValueError("audio_features must be [80, mel_frames]")
        if self.audio_feature_length <= 0:
            raise ValueError("audio_feature_length must be positive")
        if self.audio_feature_length > self.audio_features.shape[1]:
            raise ValueError("audio_feature_length exceeds feature tensor")
        if self.audio_token_length <= 2:
            raise ValueError("audio_token_length must include boundaries and frames")
        if not np.isfinite(self.audio_features).all():
            raise ValueError("audio_features contain non-finite values")
        if self.rendered_prompt.count("<|audio|>") != 1:
            raise ValueError("Rendered prompt must contain exactly one audio token")
