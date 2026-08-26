from __future__ import annotations

from typing import Any

from examples.format_following.src.metric.instruction_following_eval import (
    FormatFollowingMetric as _OfficialFormatFollowingMetric,
)
from flexeval.core.language_model.base import LMOutput


class FormatFollowingMetric(_OfficialFormatFollowingMetric):  # type: ignore[misc]
    """Bridge the pinned metric's legacy string contract to FlexEval LMOutput."""

    def evaluate(
        self,
        lm_outputs: list[str | LMOutput],
        references_list: list[list[str]],
        extra_info_list: list[dict[str, Any]] | None = None,
    ) -> Any:
        texts = [
            value.text if isinstance(value, LMOutput) else value for value in lm_outputs
        ]
        return super().evaluate(texts, references_list, extra_info_list)
