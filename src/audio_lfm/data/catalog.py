from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pyarrow.dataset as pads

from audio_lfm.data.types import CatalogAudioRecord, TargetRecord
from audio_lfm.utils.hashing import canonical_sha256


class CatalogError(ValueError):
    """Base class for catalog contract failures."""


class DuplicateAudioIdError(CatalogError):
    """An audio catalog contains a repeated stable join key."""


class DuplicateTargetIdError(CatalogError):
    """A target catalog contains a repeated target identity."""


class SplitLeakageError(CatalogError):
    """Logical train/dev/holdout partitions overlap unexpectedly."""


class MissingTargetError(CatalogError):
    """A required typed target is absent for an audio item."""


AUDIO_COLUMNS = (
    "audio_id",
    "dataset",
    "flac_sha256",
    "flac_size",
    "source_id",
    "splits",
    "target_count",
    "wds_key",
    "wds_shard",
)
TARGET_COLUMNS = (
    "audio_id",
    "target_id",
    "target_type",
    "text",
    "split",
    "source",
    "generator_model",
    "generator_revision",
    "prompt_sha256",
    "review_status",
)


def _find_catalog_path(root: Path, dataset: str, suffix: str) -> Path:
    candidates = (
        root / dataset / "16k-flac" / "parquet" / suffix,
        root
        / "CaptionStew"
        / "_webdataset"
        / dataset
        / "16k-flac"
        / "parquet"
        / suffix,
        root / "_webdataset" / dataset / "16k-flac" / "parquet" / suffix,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {suffix} catalog for {dataset!r}; checked: "
        + ", ".join(map(str, candidates))
    )


def _rows(path: Path, columns: Iterable[str]) -> list[dict[str, Any]]:
    dataset = pads.dataset(path, format="parquet")
    missing = set(columns) - set(dataset.schema.names)
    if missing:
        raise CatalogError(f"Catalog {path} lacks columns: {sorted(missing)}")
    return cast(
        list[dict[str, Any]], dataset.to_table(columns=list(columns)).to_pylist()
    )


@dataclass
class CatalogIndex:
    dataset: str
    logical_split: str
    allowed_audio_ids: frozenset[str]
    audio_by_id: dict[str, CatalogAudioRecord]
    style_captions_by_id: dict[str, tuple[TargetRecord, ...]]
    transcript_by_id: dict[str, TargetRecord | None]
    selected_shards: tuple[str, ...]
    review_status_distribution: dict[str, int]
    target_count_distribution: dict[int, int]
    split_overlap_report: dict[str, int]
    fingerprint: str

    @classmethod
    def load(
        cls,
        *,
        root: str | Path,
        dataset: str,
        logical_split: str | Sequence[str],
        target_type: str = "style_caption",
        target_provider: str = "official_target",
        review_status_allowlist: set[str] | None = None,
    ) -> CatalogIndex:
        requested_splits = (
            (logical_split,) if isinstance(logical_split, str) else tuple(logical_split)
        )
        if not requested_splits:
            raise CatalogError("At least one logical split is required")
        if dataset.startswith("ParaSpeech") and "test" in requested_splits:
            raise CatalogError("ParaSpeechCaps split 'test' is forbidden; use holdout")
        root_path = Path(root)
        audio_path = _find_catalog_path(root_path, dataset, "audio")
        if target_provider == "official_target":
            target_suffix = "targets/source=official"
        elif target_provider in {"response_overlay", "caption_expansion_overlay"}:
            target_suffix = "overlays/kind=response"
        else:
            raise CatalogError(f"Unknown target provider: {target_provider}")
        try:
            target_path = _find_catalog_path(root_path, dataset, target_suffix)
        except FileNotFoundError as error:
            if target_provider in {
                "response_overlay",
                "caption_expansion_overlay",
            }:
                raise CatalogError(
                    f"Requested response overlay is absent for {dataset!r}"
                ) from error
            raise
        audio_rows = _rows(audio_path, AUDIO_COLUMNS)
        target_rows = _rows(target_path, TARGET_COLUMNS)

        all_audio: dict[str, CatalogAudioRecord] = {}
        split_sets: defaultdict[str, set[str]] = defaultdict(set)
        for row in audio_rows:
            audio_id = str(row["audio_id"])
            if audio_id in all_audio:
                raise DuplicateAudioIdError(f"Duplicate audio_id: {audio_id}")
            splits = tuple(str(value) for value in (row["splits"] or ()))
            record = CatalogAudioRecord(
                audio_id=audio_id,
                dataset=str(row["dataset"]),
                source_id=str(row["source_id"]),
                splits=splits,
                wds_key=str(row["wds_key"]),
                wds_shard=str(row["wds_shard"]),
                flac_sha256=str(row["flac_sha256"]),
                flac_size=int(row["flac_size"]),
                target_count=int(row["target_count"]),
            )
            all_audio[audio_id] = record
            for split in splits:
                split_sets[split].add(audio_id)

        overlaps = {
            "train_base_dev": len(split_sets["train_base"] & split_sets["dev"]),
            "train_base_holdout": len(split_sets["train_base"] & split_sets["holdout"]),
            "dev_holdout": len(split_sets["dev"] & split_sets["holdout"]),
            "test_holdout": len(split_sets["test"] & split_sets["holdout"]),
        }
        forbidden = {
            key: count
            for key, count in overlaps.items()
            if count and key != "test_holdout"
        }
        if dataset.startswith("ParaSpeech") and forbidden:
            raise SplitLeakageError(f"Unexpected split overlap: {overlaps}")

        target_ids: set[str] = set()
        targets_by_audio: defaultdict[str, list[TargetRecord]] = defaultdict(list)
        review_statuses: Counter[str] = Counter()
        for row in target_rows:
            target_id = str(row["target_id"])
            if target_id in target_ids:
                raise DuplicateTargetIdError(f"Duplicate target_id: {target_id}")
            target_ids.add(target_id)
            review_status = str(row["review_status"] or "")
            review_statuses[review_status] += 1
            if (
                review_status_allowlist is not None
                and review_status not in review_status_allowlist
            ):
                continue
            target_record = TargetRecord(
                audio_id=str(row["audio_id"]),
                target_id=target_id,
                target_type=str(row["target_type"]),
                text=str(row["text"]),
                split=str(row["split"]),
                source=str(row["source"]),
                review_status=review_status,
                generator_model=(
                    str(row["generator_model"]) if row["generator_model"] else None
                ),
                generator_revision=(
                    str(row["generator_revision"])
                    if row["generator_revision"]
                    else None
                ),
                prompt_sha256=(
                    str(row["prompt_sha256"]) if row["prompt_sha256"] else None
                ),
            )
            targets_by_audio[target_record.audio_id].append(target_record)

        allowed = frozenset(
            audio_id for split in requested_splits for audio_id in split_sets[split]
        )
        audio_by_id = {audio_id: all_audio[audio_id] for audio_id in sorted(allowed)}
        styles: dict[str, tuple[TargetRecord, ...]] = {}
        transcripts: dict[str, TargetRecord | None] = {}
        for audio_id in sorted(allowed):
            records = targets_by_audio.get(audio_id, [])
            style_records = tuple(
                sorted(
                    (record for record in records if record.target_type == target_type),
                    key=lambda record: record.target_id,
                )
            )
            transcript_records = sorted(
                (record for record in records if record.target_type == "transcription"),
                key=lambda record: record.target_id,
            )
            if dataset.startswith("ParaSpeech") and not style_records:
                raise MissingTargetError(
                    f"ParaSpeech audio {audio_id!r} has no {target_type!r} target"
                )
            if len(transcript_records) > 1:
                raise CatalogError(f"Audio {audio_id!r} has multiple transcripts")
            styles[audio_id] = style_records
            transcripts[audio_id] = (
                transcript_records[0] if transcript_records else None
            )

        fingerprint_payload = {
            "audio": [record.__dict__ for record in audio_by_id.values()],
            "targets": [
                record.__dict__
                for audio_id in sorted(allowed)
                for record in targets_by_audio.get(audio_id, [])
            ],
        }
        return cls(
            dataset=dataset,
            logical_split="+".join(requested_splits),
            allowed_audio_ids=allowed,
            audio_by_id=audio_by_id,
            style_captions_by_id=styles,
            transcript_by_id=transcripts,
            selected_shards=tuple(
                sorted({record.wds_shard for record in audio_by_id.values()})
            ),
            review_status_distribution=dict(review_statuses),
            target_count_distribution=dict(
                Counter(len(styles[audio_id]) for audio_id in allowed)
            ),
            split_overlap_report=overlaps,
            fingerprint=canonical_sha256(fingerprint_payload),
        )
