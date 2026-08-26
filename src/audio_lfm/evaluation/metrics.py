from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class NllAccumulator:
    loss_sum: float = 0.0
    tokens: int = 0
    targets: int = 0
    by_audio: dict[str, list[float]] | None = None

    def __post_init__(self) -> None:
        if self.by_audio is None:
            self.by_audio = defaultdict(list)

    def add(self, *, audio_id: str, loss_sum: float, token_count: int) -> None:
        if token_count <= 0:
            raise ValueError("token_count must be positive")
        self.loss_sum += loss_sum
        self.tokens += token_count
        self.targets += 1
        assert self.by_audio is not None
        self.by_audio[audio_id].append(loss_sum / token_count)

    def results(self) -> dict[str, float | int]:
        if not self.tokens or not self.targets:
            raise RuntimeError("No NLL observations")
        assert self.by_audio is not None
        per_audio = [sum(values) / len(values) for values in self.by_audio.values()]
        return {
            "aggregate_target_nll": self.loss_sum / self.tokens,
            "target_count": self.targets,
            "token_count": self.tokens,
            "audio_weighted_mean_nll": sum(per_audio) / len(per_audio),
        }
