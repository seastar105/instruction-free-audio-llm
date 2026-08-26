from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch

from audio_lfm.data.types import CatalogAudioRecord, LocalSampleReference
from audio_lfm.utils.hashing import deterministic_int


class DataContractError(ValueError):
    """Decoded data violates the strict audio or metadata contract."""


class LongAudioSkipped(DataContractError):
    """An overlength sample was excluded by the configured skip policy."""


def _member(sample: dict[str, Any], name: str) -> bytes:
    value = sample.get(name)
    if value is None:
        value = sample.get(f".{name}")
    if not isinstance(value, bytes):
        raise DataContractError(f"Sample lacks binary {name!r} member")
    return value


def decode_webdataset_sample(
    sample: dict[str, Any],
    *,
    catalog_record: CatalogAudioRecord,
    max_audio_seconds: float,
    long_audio_policy: str,
    seed: int,
    epoch: int,
) -> tuple[torch.Tensor, int, dict[str, Any], int | None, int]:
    key = str(sample.get("__key__", ""))
    metadata = json.loads(_member(sample, "json").decode("utf-8"))
    if not isinstance(metadata, dict):
        raise DataContractError("JSON metadata must be an object")
    metadata_audio_id = str(metadata.get("audio_id", ""))
    if key != metadata_audio_id or key != catalog_record.audio_id:
        raise DataContractError(
            f"WebDataset key/JSON/catalog mismatch: {key!r}, "
            f"{metadata_audio_id!r}, {catalog_record.audio_id!r}"
        )
    flac_bytes = _member(sample, "flac")
    if len(flac_bytes) != catalog_record.flac_size:
        raise DataContractError(f"FLAC size mismatch for {catalog_record.audio_id!r}")
    if hashlib.sha256(flac_bytes).hexdigest() != catalog_record.flac_sha256:
        raise DataContractError(
            f"FLAC SHA-256 mismatch for {catalog_record.audio_id!r}"
        )
    try:
        array, sample_rate = sf.read(
            io.BytesIO(flac_bytes), dtype="float32", always_2d=True
        )
    except sf.LibsndfileError as error:
        # Some libsndfile builds fail while formatting this native exception.
        # Never stringify it; preserve a stable audio identity for worker errors.
        raise DataContractError(
            f"FLAC decode failed for {catalog_record.audio_id!r}"
        ) from error
    if sample_rate != 16_000:
        raise DataContractError(f"Expected 16000 Hz, received {sample_rate}")
    if array.shape[1] != 1:
        raise DataContractError(f"Expected one channel, received {array.shape[1]}")
    original = int(array.shape[0])
    maximum = int(max_audio_seconds * sample_rate)
    crop_start: int | None = None
    if original > maximum:
        if long_audio_policy == "skip":
            raise LongAudioSkipped(
                f"Audio {key!r} is {original / sample_rate:.3f}s, "
                f"over {max_audio_seconds}s"
            )
        if long_audio_policy == "chunk_pad":
            pass
        elif long_audio_policy == "center_crop":
            crop_start = (original - maximum) // 2
        elif long_audio_policy == "random_crop":
            crop_start = deterministic_int(
                seed=seed,
                epoch=epoch,
                audio_id=key,
                upper_exclusive=original - maximum + 1,
            )
        else:
            raise DataContractError(f"Unknown long-audio policy: {long_audio_policy}")
        if crop_start is not None:
            array = array[crop_start : crop_start + maximum]
    metadata = dict(metadata)
    metadata["audio_contract"] = {
        "sample_rate": sample_rate,
        "original_num_samples": original,
        "evaluated_num_samples": len(array),
        "original_duration_seconds": original / sample_rate,
        "crop_policy": long_audio_policy if crop_start is not None else None,
        "crop_start_sample": crop_start,
        "cropped_duration_seconds": len(array) / sample_rate,
        "chunk_seconds": max_audio_seconds
        if long_audio_policy == "chunk_pad"
        else None,
        "chunk_count": (
            (original + maximum - 1) // maximum
            if long_audio_policy == "chunk_pad"
            else None
        ),
    }
    waveform = torch.from_numpy(np.asarray(array[:, 0]).copy())
    return waveform, sample_rate, metadata, crop_start, original


class DurationSidecar:
    def __init__(self, path: str | Path, *, require_exact: bool = False) -> None:
        self.path = Path(path)
        self._records: dict[str, tuple[int, float, str]] = {}
        self._local_references: dict[str, LocalSampleReference] = {}
        if self.path.exists():
            table = pq.read_table(self.path)
            required = {"audio_id", "num_samples", "duration_seconds", "flac_sha256"}
            if not required.issubset(table.column_names):
                raise DataContractError("Duration sidecar schema is invalid")
            if require_exact:
                if "num_samples_source" not in table.column_names:
                    raise DataContractError(
                        "Exact duration sidecar lacks num_samples_source"
                    )
                sources = set(table["num_samples_source"].to_pylist())
                exact_sources = {
                    "flac_streaminfo",
                    "libsndfile_header",
                    "flac_empty",
                }
                if not sources or not sources.issubset(exact_sources):
                    raise DataContractError(
                        f"Duration sidecar contains non-exact sources: {sources}"
                    )
                reference_columns = {
                    "wds_shard",
                    "wds_key",
                    "flac_offset",
                    "flac_size",
                    "json_offset",
                    "json_size",
                }
                if not reference_columns.issubset(table.column_names):
                    raise DataContractError(
                        "Exact duration sidecar lacks local TAR byte ranges"
                    )
            for row in table.to_pylist():
                audio_id = str(row["audio_id"])
                if audio_id in self._records:
                    raise DataContractError(
                        f"Duration sidecar contains duplicate audio_id {audio_id!r}"
                    )
                self._records[audio_id] = (
                    int(row["num_samples"]),
                    float(row["duration_seconds"]),
                    str(row["flac_sha256"]),
                )
                has_local_reference = all(
                    name in row and row[name] is not None
                    for name in (
                        "wds_shard",
                        "wds_key",
                        "flac_offset",
                        "flac_size",
                        "json_offset",
                        "json_size",
                    )
                )
                if require_exact and not has_local_reference:
                    raise DataContractError(
                        f"Exact sidecar has an incomplete TAR reference for "
                        f"{audio_id!r}"
                    )
                if has_local_reference:
                    self._local_references[audio_id] = LocalSampleReference(
                        wds_shard=str(row["wds_shard"]),
                        wds_key=str(row["wds_key"]),
                        flac_offset=int(row["flac_offset"]),
                        flac_size=int(row["flac_size"]),
                        json_offset=int(row["json_offset"]),
                        json_size=int(row["json_size"]),
                    )

    def get(self, record: CatalogAudioRecord) -> tuple[int, float] | None:
        cached = self._records.get(record.audio_id)
        if cached is None:
            return None
        num_samples, duration, digest = cached
        if digest != record.flac_sha256:
            raise DataContractError(
                f"Stale duration sidecar entry for {record.audio_id!r}"
            )
        return num_samples, duration

    def local_reference(
        self, record: CatalogAudioRecord
    ) -> LocalSampleReference | None:
        reference = self._local_references.get(record.audio_id)
        if reference is None:
            return None
        if (
            reference.wds_shard != record.wds_shard
            or reference.wds_key != record.wds_key
            or reference.flac_size != record.flac_size
        ):
            raise DataContractError(
                f"Stale local TAR reference for {record.audio_id!r}"
            )
        return reference

    def update(self, record: CatalogAudioRecord, num_samples: int) -> None:
        self._records[record.audio_id] = (
            num_samples,
            num_samples / 16_000,
            record.flac_sha256,
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "audio_id": audio_id,
                "num_samples": value[0],
                "duration_seconds": value[1],
                "flac_sha256": value[2],
            }
            for audio_id, value in sorted(self._records.items())
        ]
        pq.write_table(pa.Table.from_pylist(rows), self.path)
