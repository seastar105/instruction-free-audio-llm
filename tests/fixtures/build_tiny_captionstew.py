from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf


def _flac_bytes(
    *, sample_rate: int = 16_000, channels: int = 1, seconds: float = 0.08
) -> bytes:
    samples = max(1, int(sample_rate * seconds))
    time = np.arange(samples, dtype=np.float32) / sample_rate
    wave = 0.1 * np.sin(2 * np.pi * 220 * time)
    if channels > 1:
        wave = np.stack([wave, wave * 0.9], axis=1)
    buffer = io.BytesIO()
    sf.write(buffer, wave, sample_rate, format="FLAC", subtype="PCM_16")
    return buffer.getvalue()


def _add_tar_member(archive: tarfile.TarFile, name: str, value: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(value)
    archive.addfile(info, io.BytesIO(value))


def build_tiny_captionstew(
    root: Path,
    *,
    malformed: str | None = None,
) -> dict[str, Any]:
    dataset = "ParaSpeechCaps-Base"
    base = root / "_webdataset" / dataset / "16k-flac"
    audio_dir = base / "parquet" / "audio"
    target_dir = base / "parquet" / "targets" / "source=official"
    shard_dir = base / "shards"
    audio_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)
    shard_dir.mkdir(parents=True)
    specs = [
        ("train-0", ("train_base",)),
        ("train-1", ("train_base",)),
        ("dev-0", ("dev",)),
        ("holdout-0", ("holdout", "test")),
    ]
    if malformed == "overlap":
        specs[0] = ("train-0", ("train_base", "dev"))
    audio_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for index, (audio_id, splits) in enumerate(specs):
        rate = 8_000 if malformed == "wrong_rate" and index == 0 else 16_000
        channels = 2 if malformed == "stereo" and index == 0 else 1
        flac = _flac_bytes(sample_rate=rate, channels=channels)
        if malformed == "invalid_flac" and index == 0:
            flac = b"not-a-valid-flac-stream"
        key = (
            "wrong-json-id" if malformed == "key_mismatch" and index == 0 else audio_id
        )
        styles = [
            {
                "audio_id": audio_id,
                "target_id": f"{audio_id}-style-0",
                "target_type": "style_caption",
                "text": f"A calm voice number {index}.",
                "split": splits[0],
                "source": "official",
                "generator_model": None,
                "generator_revision": None,
                "prompt_sha256": None,
                "review_status": "accepted",
            }
        ]
        if audio_id == "train-0":
            styles.append(
                {
                    **styles[0],
                    "target_id": f"{audio_id}-style-1",
                    "text": "A second calm style reference.",
                }
            )
        if malformed == "missing_style" and index == 0:
            styles = []
        transcript = {
            "audio_id": audio_id,
            "target_id": f"{audio_id}-transcript",
            "target_type": "transcription",
            "text": f"spoken words {index}",
            "split": splits[0],
            "source": "official",
            "generator_model": None,
            "generator_revision": None,
            "prompt_sha256": None,
            "review_status": "accepted",
        }
        targets = [*styles, transcript]
        metadata = {
            "audio_id": key,
            "dataset": dataset,
            "targets": targets,
            "license": "synthetic-test-only",
        }
        audio_rows.append(
            {
                "audio_id": audio_id,
                "dataset": dataset,
                "flac_sha256": hashlib.sha256(flac).hexdigest(),
                "flac_size": len(flac),
                "source_id": f"source-{index}",
                "splits": list(splits),
                "target_count": len(targets),
                "wds_key": audio_id,
                "wds_shard": (f"_webdataset/{dataset}/16k-flac/shards/tiny-000000.tar"),
            }
        )
        target_rows.extend(targets)
        samples.append(
            {
                "__key__": audio_id,
                "flac": flac,
                "json": json.dumps(metadata).encode(),
            }
        )
    if malformed == "duplicate_audio":
        audio_rows.append(dict(audio_rows[0]))
    pq.write_table(pa.Table.from_pylist(audio_rows), audio_dir / "part.parquet")
    pq.write_table(pa.Table.from_pylist(target_rows), target_dir / "part.parquet")
    tar_path = shard_dir / "tiny-000000.tar"
    with tarfile.open(tar_path, "w") as archive:
        for sample in samples:
            key = sample["__key__"]
            _add_tar_member(archive, f"{key}.flac", sample["flac"])
            _add_tar_member(archive, f"{key}.json", sample["json"])
    return {
        "root": root,
        "dataset": dataset,
        "tar_path": tar_path,
        "samples": samples,
    }
