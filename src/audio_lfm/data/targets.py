from __future__ import annotations

from collections.abc import Sequence

from audio_lfm.data.types import TargetRecord
from audio_lfm.utils.hashing import stable_reference_index


def select_target(
    references: Sequence[TargetRecord], *, seed: int, epoch: int, audio_id: str
) -> TargetRecord:
    if not references:
        raise ValueError(f"Audio {audio_id!r} has no target references")
    ordered = sorted(references, key=lambda target: target.target_id)
    index = stable_reference_index(
        seed=seed,
        epoch=epoch,
        audio_id=audio_id,
        num_references=len(ordered),
    )
    return ordered[index]


def validate_target_consistency(
    metadata: dict[str, object], parquet_targets: Sequence[TargetRecord]
) -> None:
    raw_targets = metadata.get("targets")
    if raw_targets is None:
        return
    if not isinstance(raw_targets, list):
        raise ValueError("JSON metadata targets must be a list")
    json_identity = {
        (
            str(item.get("target_id")),
            str(item.get("target_type")),
            str(item.get("text")),
        )
        for item in raw_targets
        if isinstance(item, dict)
    }
    parquet_identity = {
        (target.target_id, target.target_type, target.text)
        for target in parquet_targets
    }
    if json_identity != parquet_identity:
        raise ValueError("JSON and Parquet target records do not match")
