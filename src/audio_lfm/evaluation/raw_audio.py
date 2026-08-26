from __future__ import annotations

import numpy as np

from audio_lfm.data.decode import DataContractError


def validate_raw_audio(
    waveform: np.ndarray, sample_rate: int, *, max_audio_seconds: float
) -> np.ndarray:
    if waveform.ndim != 1:
        raise DataContractError("Expected mono 1D audio")
    if sample_rate != 16_000:
        raise DataContractError("Expected 16 kHz audio")
    if waveform.size == 0:
        raise DataContractError("Empty waveform")
    if waveform.size > int(max_audio_seconds * sample_rate):
        raise DataContractError("Audio exceeds configured maximum")
    if not np.isfinite(waveform).all():
        raise DataContractError("Audio contains NaN or Inf")
    return waveform.astype(np.float32, copy=False)
