from __future__ import annotations

import importlib.util
import os

import pytest


@pytest.mark.private_data
@pytest.mark.skipif(not os.environ.get("HF_TOKEN"), reason="HF_TOKEN is unavailable")
@pytest.mark.skipif(
    not os.environ.get("CAPTIONSTEW_ROOT"), reason="CAPTIONSTEW_ROOT is unavailable"
)
@pytest.mark.skipif(
    importlib.util.find_spec("captionstew") is None,
    reason="CaptionStew training client is unavailable",
)
def test_private_bucket_first_hundred_samples() -> None:
    from itertools import islice

    from audio_lfm.cli import _backend, _catalog
    from audio_lfm.config import load_config

    config = load_config("configs/paraspeech_whisper_lfm2.yaml")
    catalog = _catalog(config, "train_base")
    backend = _backend(config, catalog)
    backend.strict_target_consistency = True
    examples = list(islice(backend.iter_epoch(0), 100))
    assert len(examples) == 100
    assert len({example.audio_id for example in examples}) == 100
    assert all(example.sample_rate == 16_000 for example in examples)
    assert all(example.waveform.ndim == 1 for example in examples)
    assert all(example.audio_id in catalog.allowed_audio_ids for example in examples)
