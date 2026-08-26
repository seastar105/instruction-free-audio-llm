from __future__ import annotations

from pathlib import Path

import pytest
import webdataset as wds

from audio_lfm.data.captionstew_backend import CaptionStewBackend
from audio_lfm.data.catalog import CatalogIndex
from audio_lfm.data.decode import DataContractError
from tests.fixtures.build_tiny_captionstew import build_tiny_captionstew


def _stream(fixture: dict[str, object]) -> list[object]:
    index = CatalogIndex.load(
        root=fixture["root"],
        dataset="ParaSpeechCaps-Base",
        logical_split="train_base",
    )
    backend = CaptionStewBackend(
        captionstew_root=fixture["root"],
        dataset="ParaSpeechCaps-Base",
        catalog=index,
        shard_shuffle=0,
        sample_shuffle=0,
        max_audio_seconds=30,
        long_audio_policy="skip",
        strict_target_consistency=True,
        local_samples=fixture["samples"],
    )
    return list(backend.iter_epoch(0))


def test_synthetic_stream_retains_provenance(
    tiny_captionstew: dict[str, object],
) -> None:
    examples = _stream(tiny_captionstew)
    assert [example.audio_id for example in examples] == ["train-0", "train-1"]
    assert examples[0].sample_rate == 16_000
    assert examples[0].waveform.ndim == 1
    assert examples[0].metadata["license"] == "synthetic-test-only"
    assert "spoken words" not in examples[0].selected_target.text


def test_uncompressed_tar_streams_through_webdataset(
    tiny_captionstew: dict[str, object],
) -> None:
    index = CatalogIndex.load(
        root=tiny_captionstew["root"],
        dataset="ParaSpeechCaps-Base",
        logical_split="dev",
    )
    tar_stream = wds.WebDataset(str(tiny_captionstew["tar_path"]), shardshuffle=False)
    backend = CaptionStewBackend(
        captionstew_root=tiny_captionstew["root"],
        dataset="ParaSpeechCaps-Base",
        catalog=index,
        shard_shuffle=0,
        sample_shuffle=0,
        max_audio_seconds=30,
        long_audio_policy="skip",
        strict_target_consistency=True,
        local_samples=tar_stream,
    )
    examples = list(backend.iter_epoch(0))
    assert [example.audio_id for example in examples] == ["dev-0"]


@pytest.mark.parametrize(
    "malformed", ["wrong_rate", "stereo", "key_mismatch", "invalid_flac"]
)
def test_audio_contract_failures(tmp_path: Path, malformed: str) -> None:
    with pytest.raises(DataContractError):
        _stream(build_tiny_captionstew(tmp_path, malformed=malformed))


def test_invalid_flac_failure_retains_audio_identity(tmp_path: Path) -> None:
    with pytest.raises(DataContractError, match="train-0.*failed validation"):
        _stream(build_tiny_captionstew(tmp_path, malformed="invalid_flac"))
