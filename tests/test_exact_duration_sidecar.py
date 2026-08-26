from __future__ import annotations

import io
from pathlib import Path

import pytest
import soundfile as sf

from scripts.build_exact_duration_sidecar import _exact_samples, _streaminfo


def test_flac_streaminfo_returns_exact_sample_count(
    tiny_captionstew: dict[str, object],
) -> None:
    samples = tiny_captionstew["samples"]
    assert isinstance(samples, list)
    flac = samples[0]["flac"]
    assert isinstance(flac, bytes)
    rate, channels, count = _streaminfo(flac[:42], audio_id="train-0")
    with sf.SoundFile(io.BytesIO(flac)) as audio:
        assert (rate, channels, count) == (
            audio.samplerate,
            audio.channels,
            audio.frames,
        )


def test_flac_streaminfo_rejects_invalid_header() -> None:
    with pytest.raises(ValueError, match="Invalid FLAC header"):
        _streaminfo(b"not-flac", audio_id="broken")


def test_streaminfo_allows_spec_defined_unknown_total_samples(
    tiny_captionstew: dict[str, object],
) -> None:
    samples = tiny_captionstew["samples"]
    assert isinstance(samples, list)
    header = bytearray(samples[0]["flac"][:42])
    packed = int.from_bytes(header[18:26], "big")
    header[18:26] = (packed & ~((1 << 36) - 1)).to_bytes(8, "big")
    rate, channels, count = _streaminfo(bytes(header), audio_id="unknown-length")
    assert (rate, channels, count) == (16_000, 1, 0)


def test_exact_scan_records_lazy_tar_byte_ranges(
    tiny_captionstew: dict[str, object],
) -> None:
    root = tiny_captionstew["root"]
    samples = tiny_captionstew["samples"]
    assert isinstance(root, Path)
    assert isinstance(samples, list)
    by_id = {str(sample["__key__"]): sample for sample in samples}
    references = _exact_samples(root, "ParaSpeechCaps-Base", set(by_id))
    assert set(references) == set(by_id)
    for audio_id, reference in references.items():
        shard = root / str(reference["wds_shard"])
        with shard.open("rb") as stream:
            stream.seek(int(reference["flac_offset"]))
            flac = stream.read(int(reference["flac_size"]))
            stream.seek(int(reference["json_offset"]))
            metadata = stream.read(int(reference["json_size"]))
        assert flac == by_id[audio_id]["flac"]
        assert metadata == by_id[audio_id]["json"]
