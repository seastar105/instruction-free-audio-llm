from __future__ import annotations

import base64
import hashlib
import io
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SAMPLE_RATE = 16_000

_FORMAT_SUFFIXES = {
    "AIFF": ".aiff",
    "AU": ".au",
    "CAF": ".caf",
    "FLAC": ".flac",
    "MP3": ".mp3",
    "MPEG": ".mp3",
    "OGG": ".ogg",
    "RF64": ".wav",
    "W64": ".w64",
    "WAV": ".wav",
    "WAVEX": ".wav",
}


def _resolve_path(value: str, root: Path) -> Path:
    candidate = Path(value)
    candidates = [candidate, root / candidate, root / str(candidate).lstrip("./")]
    for path in candidates:
        if path.is_file():
            return path
    matches = list(root.rglob(candidate.name))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous audio basename {candidate.name!r} under {root}")
    raise FileNotFoundError(f"Audio file {value!r} was not found under {root}")


def _bytes_or_path(value: object, root: Path) -> bytes | Path:
    if isinstance(value, bytes):
        return value
    if isinstance(value, (str, Path)):
        return _resolve_path(str(value), root)
    if isinstance(value, dict):
        raw = value.get("bytes")
        if isinstance(raw, bytes):
            return raw
        path = value.get("path")
        if isinstance(path, str):
            return _resolve_path(path, root)
    raise TypeError(f"Unsupported audio value type: {type(value).__name__}")


def _path_or_bytes(value: object, root: Path) -> Path | bytes:
    """Prefer an existing dataset file; fall back to embedded Parquet bytes."""
    if isinstance(value, (str, Path)):
        return _resolve_path(str(value), root)
    if isinstance(value, bytes):
        return value
    if isinstance(value, dict):
        path = value.get("path")
        if isinstance(path, str):
            try:
                return _resolve_path(path, root)
            except FileNotFoundError:
                pass
        raw = value.get("bytes")
        if isinstance(raw, bytes):
            return raw
    raise TypeError(f"Unsupported audio value type: {type(value).__name__}")


def _audio_metadata(
    source: bytes | Path, *, max_seconds: float
) -> tuple[Any, dict[str, Any]]:
    stream = io.BytesIO(source) if isinstance(source, bytes) else source
    info = sf.info(stream)
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError("Audio header reports an empty or invalid waveform")
    duration = int(info.frames) / int(info.samplerate)
    if duration > max_seconds + 1e-6:
        raise ValueError(
            f"audio duration {duration:.3f}s exceeds the configured "
            f"{max_seconds:.3f}s per-item evaluation limit"
        )
    metadata = {
        "original_sample_rate": int(info.samplerate),
        "original_channels": int(info.channels),
        "evaluated_sample_rate": TARGET_SAMPLE_RATE,
        "evaluated_num_samples": math.ceil(
            int(info.frames) * TARGET_SAMPLE_RATE / int(info.samplerate)
        ),
        "duration_seconds": duration,
        "whisper_chunk_count": max(1, math.ceil(duration / 30.0 - 1e-12)),
    }
    return info, metadata


def _materialize_embedded_audio(raw: bytes, cache_root: Path, suffix: str) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(raw).hexdigest()
    destination = cache_root / f"{digest}{suffix}"
    if destination.exists():
        return destination
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=cache_root,
            prefix=f".{digest}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def audio_file_uri(
    value: object,
    root: str | Path,
    *,
    max_seconds: float,
    cache_directory: str = ".vllm-media-cache",
) -> tuple[str, dict[str, Any]]:
    """Return an allowlist-contained file URI without decoding the waveform."""
    allowed_root = Path(root).resolve()
    source = _path_or_bytes(value, allowed_root)
    info, metadata = _audio_metadata(source, max_seconds=max_seconds)
    if isinstance(source, bytes):
        suffix = _FORMAT_SUFFIXES.get(str(info.format).upper())
        if suffix is None:
            raise ValueError(
                f"Unsupported embedded audio format for file transport: {info.format}"
            )
        path = _materialize_embedded_audio(
            source, allowed_root / cache_directory, suffix
        )
        metadata["materialized_from_embedded_bytes"] = True
    else:
        path = source.resolve()
        metadata["materialized_from_embedded_bytes"] = False
    if not path.is_relative_to(allowed_root):
        raise ValueError(f"Audio path {path} escapes allowlisted root {allowed_root}")
    metadata["transport"] = "file-uri"
    return path.as_uri(), metadata


def audio_num_samples(value: object, root: str | Path) -> int:
    """Read only the audio header and conservatively estimate 16 kHz samples."""
    source = _bytes_or_path(value, Path(root))
    stream = io.BytesIO(source) if isinstance(source, bytes) else source
    info = sf.info(stream)
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError("Audio header reports an empty or invalid waveform")
    return math.ceil(int(info.frames) * TARGET_SAMPLE_RATE / int(info.samplerate))


def projected_audio_tokens(
    num_samples: int, *, chunk_samples: int, stack_factor: int
) -> int:
    """Match training's chunked Whisper and projector length formulas exactly."""
    if num_samples <= 0 or chunk_samples <= 0 or stack_factor <= 0:
        raise ValueError("Audio/chunk lengths and stack factor must be positive")
    full_chunks, remainder = divmod(num_samples, chunk_samples)

    def encoder_frames(samples: int) -> int:
        mel_frames = (samples + 159) // 160
        return (mel_frames + 1) // 2

    total_encoder_frames = full_chunks * encoder_frames(chunk_samples)
    if remainder:
        total_encoder_frames += encoder_frames(remainder)
    projected_frames = (total_encoder_frames + stack_factor - 1) // stack_factor
    return projected_frames + 2


def audio_data_uri(
    value: object, root: str | Path, *, max_seconds: float
) -> tuple[str, dict[str, Any]]:
    source = _bytes_or_path(value, Path(root))
    stream = io.BytesIO(source) if isinstance(source, bytes) else source
    with sf.SoundFile(stream) as handle:
        original_rate = int(handle.samplerate)
        original_channels = int(handle.channels)
        audio = handle.read(dtype="float32", always_2d=True)
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains non-finite samples")
    mono = audio.mean(axis=1, dtype=np.float32)
    if original_rate != TARGET_SAMPLE_RATE:
        divisor = math.gcd(original_rate, TARGET_SAMPLE_RATE)
        mono = resample_poly(
            mono, TARGET_SAMPLE_RATE // divisor, original_rate // divisor
        ).astype(np.float32, copy=False)
    duration = len(mono) / TARGET_SAMPLE_RATE
    if duration > max_seconds + 1e-6:
        raise ValueError(
            f"audio duration {duration:.3f}s exceeds the configured "
            f"{max_seconds:.3f}s per-item evaluation limit"
        )
    encoded = io.BytesIO()
    sf.write(encoded, mono, TARGET_SAMPLE_RATE, format="WAV", subtype="PCM_16")
    payload = base64.b64encode(encoded.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{payload}", {
        "original_sample_rate": original_rate,
        "original_channels": original_channels,
        "evaluated_sample_rate": TARGET_SAMPLE_RATE,
        "evaluated_num_samples": len(mono),
        "duration_seconds": duration,
        "whisper_chunk_count": max(1, math.ceil(duration / 30.0 - 1e-12)),
    }
