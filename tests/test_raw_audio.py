from __future__ import annotations

import numpy as np
import pytest

from audio_lfm.data.decode import DataContractError
from audio_lfm.evaluation.raw_audio import validate_raw_audio


@pytest.mark.parametrize(
    ("waveform", "rate"),
    [
        (np.zeros((10, 2), np.float32), 16_000),
        (np.zeros(10, np.float32), 8_000),
        (np.zeros(0, np.float32), 16_000),
        (np.array([np.nan], np.float32), 16_000),
    ],
)
def test_strict_raw_audio_rejects_invalid(waveform, rate) -> None:
    with pytest.raises(DataContractError):
        validate_raw_audio(waveform, rate, max_audio_seconds=30)
