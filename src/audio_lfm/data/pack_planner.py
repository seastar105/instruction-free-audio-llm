from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Generic

from audio_lfm.data.types import PackableT, PackPlan


class OversizedExampleError(ValueError):
    """A logical example cannot fit and must not be truncated."""


class OnlinePackPlanner(Generic[PackableT]):
    def __init__(
        self,
        *,
        max_lfm_tokens: int,
        max_sample_lfm_tokens: int | None = None,
        planning_buffer_examples: int,
        max_examples_per_pack: int | None,
        oversized_example_policy: str = "error",
        best_fit_decreasing: bool = True,
    ) -> None:
        if min(max_lfm_tokens, planning_buffer_examples) <= 0:
            raise ValueError("Packing limits must be positive")
        if max_examples_per_pack is not None and max_examples_per_pack <= 0:
            raise ValueError("max_examples_per_pack must be positive or null")
        if oversized_example_policy not in {"error", "skip"}:
            raise ValueError("oversized_example_policy must be 'error' or 'skip'")
        self.max_lfm_tokens = max_lfm_tokens
        self.max_sample_lfm_tokens = max_sample_lfm_tokens or max_lfm_tokens
        if self.max_sample_lfm_tokens > self.max_lfm_tokens:
            raise ValueError(
                "The per-sample limit cannot exceed the packed batch limit"
            )
        self.planning_buffer_examples = planning_buffer_examples
        self.max_examples_per_pack = max_examples_per_pack
        self.oversized_example_policy = oversized_example_policy
        self.best_fit_decreasing = best_fit_decreasing
        self.oversized_example_count = 0

    def plan(self, examples: Iterable[PackableT]) -> Iterator[PackPlan[PackableT]]:
        iterator = iter(examples)
        while True:
            buffer: list[PackableT] = []
            for _ in range(self.planning_buffer_examples):
                try:
                    buffer.append(next(iterator))
                except StopIteration:
                    break
            if not buffer:
                return
            yield from self.pack_buffer(buffer)

    def pack_buffer(self, buffer: list[PackableT]) -> list[PackPlan[PackableT]]:
        eligible: list[PackableT] = []
        for example in buffer:
            if example.estimated_total_lfm_length > self.max_sample_lfm_tokens:
                if self.oversized_example_policy == "error":
                    raise OversizedExampleError(
                        f"Audio {example.audio_id!r} needs "
                        f"{example.estimated_total_lfm_length} tokens, limit is "
                        f"{self.max_sample_lfm_tokens}; audio is never truncated"
                    )
                self.oversized_example_count += 1
                continue
            eligible.append(example)
        indexed = list(enumerate(eligible))
        if self.best_fit_decreasing:
            indexed.sort(
                key=lambda item: (-item[1].estimated_total_lfm_length, item[0])
            )
        bins: list[tuple[int, list[tuple[int, PackableT]]]] = []
        for original_index, example in indexed:
            size = example.estimated_total_lfm_length
            candidates = [
                (self.max_lfm_tokens - used - size, bin_index)
                for bin_index, (used, values) in enumerate(bins)
                if (
                    self.max_examples_per_pack is None
                    or len(values) < self.max_examples_per_pack
                )
                and used + size <= self.max_lfm_tokens
            ]
            if candidates:
                _, bin_index = min(candidates)
                used, values = bins[bin_index]
                values.append((original_index, example))
                bins[bin_index] = (used + size, values)
            else:
                bins.append((size, [(original_index, example)]))
        bins.sort(key=lambda item: min(index for index, _ in item[1]))
        return [
            PackPlan(
                examples=[example for _, example in values],
                estimated_total_lfm_length=used,
            )
            for used, values in bins
        ]

    def utilization(self, plan: PackPlan[PackableT]) -> float:
        return plan.estimated_total_lfm_length / self.max_lfm_tokens
