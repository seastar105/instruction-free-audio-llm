from __future__ import annotations

from pathlib import Path

import pytest

from audio_lfm.data.catalog import (
    CatalogIndex,
    DuplicateAudioIdError,
    MissingTargetError,
    SplitLeakageError,
)
from tests.fixtures.build_tiny_captionstew import build_tiny_captionstew


def test_catalog_joins_and_preserves_target_typing(
    tiny_captionstew: dict[str, object],
) -> None:
    index = CatalogIndex.load(
        root=tiny_captionstew["root"],
        dataset="ParaSpeechCaps-Base",
        logical_split="train_base",
    )
    assert index.allowed_audio_ids == {"train-0", "train-1"}
    assert len(index.style_captions_by_id["train-0"]) == 2
    assert index.transcript_by_id["train-0"].target_type == "transcription"
    assert index.split_overlap_report["test_holdout"] == 1


def test_catalog_can_union_logical_splits(
    tiny_captionstew: dict[str, object],
) -> None:
    index = CatalogIndex.load(
        root=tiny_captionstew["root"],
        dataset="ParaSpeechCaps-Base",
        logical_split=["train_base", "dev"],
    )
    assert index.logical_split == "train_base+dev"
    assert index.allowed_audio_ids == {"train-0", "train-1", "dev-0"}


@pytest.mark.parametrize(
    ("malformed", "error"),
    [
        ("duplicate_audio", DuplicateAudioIdError),
        ("missing_style", MissingTargetError),
        ("overlap", SplitLeakageError),
    ],
)
def test_malformed_catalogs_fail_specifically(
    tmp_path: Path, malformed: str, error: type[Exception]
) -> None:
    fixture = build_tiny_captionstew(tmp_path, malformed=malformed)
    with pytest.raises(error):
        CatalogIndex.load(
            root=fixture["root"],
            dataset="ParaSpeechCaps-Base",
            logical_split="train_base",
        )


def test_test_split_rejected(tiny_captionstew: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="test"):
        CatalogIndex.load(
            root=tiny_captionstew["root"],
            dataset="ParaSpeechCaps-Base",
            logical_split="test",
        )
