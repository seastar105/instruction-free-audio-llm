import base64
import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_lfm_eval.audio import (
    audio_data_uri,
    audio_file_uri,
    audio_num_samples,
    projected_audio_tokens,
)


def _wav_bytes(seconds: float, sample_rate: int = 8_000) -> bytes:
    samples = np.zeros((round(seconds * sample_rate), 2), dtype=np.float32)
    stream = io.BytesIO()
    sf.write(stream, samples, sample_rate, format="WAV")
    return stream.getvalue()


def test_audio_is_normalized_to_pcm16_16khz_mono(tmp_path) -> None:
    uri, metadata = audio_data_uri(
        {"bytes": _wav_bytes(0.25), "path": None}, tmp_path, max_seconds=30.0
    )
    assert uri.startswith("data:audio/wav;base64,")
    raw = base64.b64decode(uri.partition(",")[2])
    with sf.SoundFile(io.BytesIO(raw)) as handle:
        assert handle.samplerate == 16_000
        assert handle.channels == 1
        assert handle.subtype == "PCM_16"
    assert metadata["original_channels"] == 2
    assert metadata["evaluated_num_samples"] == 4_000


def test_long_audio_is_preserved_and_chunk_count_is_reported(tmp_path) -> None:
    uri, metadata = audio_data_uri(_wav_bytes(61.0), tmp_path, max_seconds=600.0)
    assert uri.startswith("data:audio/wav;base64,")
    assert metadata["evaluated_num_samples"] == 61 * 16_000
    assert metadata["whisper_chunk_count"] == 3


def test_configured_total_audio_limit_fails_explicitly(tmp_path) -> None:
    with pytest.raises(ValueError, match="per-item evaluation limit"):
        audio_data_uri(_wav_bytes(1.1), tmp_path, max_seconds=1.0)


def test_header_length_and_chunked_token_formula(tmp_path) -> None:
    path = tmp_path / "audio.wav"
    path.write_bytes(_wav_bytes(45.0))
    assert audio_num_samples(path, tmp_path) == 45 * 16_000
    assert (
        projected_audio_tokens(45 * 16_000, chunk_samples=30 * 16_000, stack_factor=4)
        == 565
    )


def test_path_audio_is_sent_as_an_allowlisted_file_uri(tmp_path) -> None:
    path = tmp_path / "audio.wav"
    path.write_bytes(_wav_bytes(0.25, sample_rate=16_000))
    uri, metadata = audio_file_uri(path, tmp_path, max_seconds=30.0)
    assert uri == path.resolve().as_uri()
    assert metadata["transport"] == "file-uri"
    assert metadata["materialized_from_embedded_bytes"] is False


def test_embedded_audio_is_materialized_once_by_content_hash(tmp_path) -> None:
    raw = _wav_bytes(0.25, sample_rate=16_000)
    first_uri, first_metadata = audio_file_uri(raw, tmp_path, max_seconds=30.0)
    second_uri, _ = audio_file_uri(raw, tmp_path, max_seconds=30.0)
    assert first_uri == second_uri
    assert first_metadata["materialized_from_embedded_bytes"] is True
    assert len(list((tmp_path / ".vllm-media-cache").glob("*.wav"))) == 1


def test_mp3_format_label_has_a_file_transport_suffix() -> None:
    from audio_lfm_eval.audio import _FORMAT_SUFFIXES

    assert _FORMAT_SUFFIXES["MP3"] == ".mp3"


def test_file_uri_rejects_paths_outside_allowlisted_root(tmp_path) -> None:
    outside = tmp_path.parent / "outside.wav"
    outside.write_bytes(_wav_bytes(0.25, sample_rate=16_000))
    try:
        with pytest.raises(ValueError, match="escapes allowlisted root"):
            audio_file_uri(outside, tmp_path, max_seconds=30.0)
    finally:
        Path(outside).unlink()
