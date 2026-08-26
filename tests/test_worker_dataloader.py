from __future__ import annotations

import os

import pytest

from audio_lfm.data.captionstew_backend import CaptionStewBackend
from audio_lfm.data.catalog import CatalogIndex
from audio_lfm.data.loader import build_epoch_dataloader
from audio_lfm.data.mixed_backend import MixedCaptionStewBackend
from audio_lfm.data.worker_preprocessing import AudioPreprocessingConfig


def test_worker_loader_streams_each_local_sample_once(
    tiny_captionstew: dict[str, object],
) -> None:
    if os.environ.get("AUDIO_LFM_TEST_MULTIPROCESSING") != "1":
        pytest.skip("requires multiprocessing sockets outside the file sandbox")
    catalog = CatalogIndex.load(
        root=tiny_captionstew["root"],
        dataset="ParaSpeechCaps-Base",
        logical_split="train_base",
    )
    backend = CaptionStewBackend(
        captionstew_root=tiny_captionstew["root"],
        dataset="ParaSpeechCaps-Base",
        catalog=catalog,
        shard_shuffle=0,
        sample_shuffle=0,
        max_audio_seconds=30,
        long_audio_policy="skip",
        local_samples=tiny_captionstew["samples"],  # type: ignore[arg-type]
    )
    mixed = MixedCaptionStewBackend((backend,), seed=1337)
    loader = build_epoch_dataloader(
        mixed,
        epoch=0,
        num_workers=2,
        persistent_workers=False,
        prefetch_factor=2,
        preprocessing=AudioPreprocessingConfig(model_id=None, revision=None),
    )
    examples = list(loader)
    assert sorted(example.audio_id for example in examples) == ["train-0", "train-1"]
    assert all(not hasattr(example, "waveform") for example in examples)
