from __future__ import annotations

from audio_lfm.data.targets import select_target
from audio_lfm.data.types import TargetRecord
from audio_lfm.utils.hashing import stable_reference_index


def _target(index: int) -> TargetRecord:
    return TargetRecord(
        audio_id="audio",
        target_id=f"target-{index}",
        target_type="style_caption",
        text=str(index),
        split="train_base",
        source="official",
        review_status="accepted",
    )


def test_reference_choice_is_stable_and_sorted() -> None:
    targets = [_target(2), _target(0), _target(1)]
    first = select_target(targets, seed=1337, epoch=2, audio_id="audio")
    second = select_target(reversed(targets), seed=1337, epoch=2, audio_id="audio")
    assert first == second
    assert (
        first
        == targets[[target.target_id for target in targets].index(first.target_id)]
    )


def test_reference_choice_changes_across_some_epochs() -> None:
    values = {
        stable_reference_index(
            seed=1, epoch=epoch, audio_id=f"audio-{epoch % 3}", num_references=3
        )
        for epoch in range(20)
    }
    assert len(values) > 1
