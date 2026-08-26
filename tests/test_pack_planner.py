from __future__ import annotations

import pytest
import torch

from audio_lfm.data.pack_planner import OnlinePackPlanner, OversizedExampleError
from audio_lfm.data.types import (
    PrecomputedAudioExample,
    PreparedExample,
    PreparedText,
    TargetRecord,
)


def _example(index: int, length: int) -> PreparedExample:
    target = TargetRecord(
        audio_id=str(index),
        target_id=f"target-{index}",
        target_type="style_caption",
        text="x",
        split="train_base",
        source="official",
        review_status="accepted",
    )
    raw = PrecomputedAudioExample(
        audio_id=str(index),
        input_features=torch.zeros(1, 80, 10),
        effective_encoder_lengths=(1,),
        evaluated_num_samples=100,
        sample_rate=16_000,
        source_id="source",
        splits=("train_base",),
        style_captions=(target,),
        transcript=None,
        selected_target=target,
        metadata={},
        crop_start_sample=None,
        original_num_samples=100,
    )
    text = PreparedText((1,), (2,), (3,), target.target_id, "hash")
    return PreparedExample(raw, text, 1, length)


def test_planner_is_deterministic_and_emits_once() -> None:
    planner = OnlinePackPlanner(
        max_lfm_tokens=10,
        planning_buffer_examples=8,
        max_examples_per_pack=2,
    )
    examples = [_example(0, 6), _example(1, 4), _example(2, 5)]
    first = list(planner.plan(examples))
    second = list(planner.plan(examples))
    assert [[x.raw.audio_id for x in plan.examples] for plan in first] == [
        [x.raw.audio_id for x in plan.examples] for plan in second
    ]
    assert sorted(x.raw.audio_id for plan in first for x in plan.examples) == [
        "0",
        "1",
        "2",
    ]
    assert all(plan.estimated_total_lfm_length <= 10 for plan in first)


def test_oversized_singleton_fails() -> None:
    planner = OnlinePackPlanner(
        max_lfm_tokens=10,
        planning_buffer_examples=2,
        max_examples_per_pack=2,
    )
    with pytest.raises(OversizedExampleError, match="never truncated"):
        list(planner.plan([_example(0, 11)]))


def test_oversized_singleton_can_be_skipped_without_truncation() -> None:
    planner = OnlinePackPlanner(
        max_lfm_tokens=10,
        planning_buffer_examples=2,
        max_examples_per_pack=None,
        oversized_example_policy="skip",
    )
    plans = list(planner.plan([_example(0, 11), _example(1, 10)]))
    assert [[item.raw.audio_id for item in plan.examples] for plan in plans] == [["1"]]
    assert planner.oversized_example_count == 1


def test_sample_and_packed_batch_limits_are_independent() -> None:
    planner = OnlinePackPlanner(
        max_lfm_tokens=10,
        max_sample_lfm_tokens=6,
        planning_buffer_examples=4,
        max_examples_per_pack=4,
    )
    plans = list(planner.plan([_example(0, 6), _example(1, 4)]))
    assert len(plans) == 1
    assert plans[0].estimated_total_lfm_length == 10
    with pytest.raises(OversizedExampleError):
        list(planner.plan([_example(2, 7)]))


def test_planner_can_leave_example_count_uncapped() -> None:
    planner = OnlinePackPlanner(
        max_lfm_tokens=10,
        planning_buffer_examples=8,
        max_examples_per_pack=None,
    )
    plans = list(planner.plan([_example(index, 2) for index in range(5)]))
    assert len(plans) == 1
    assert len(plans[0].examples) == 5
