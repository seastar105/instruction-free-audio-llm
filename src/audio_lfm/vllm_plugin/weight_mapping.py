from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch

EXPLICIT_SKIPS = (
    "audio_tower.model.decoder.",
    "audio_tower.proj_out.",
)


def map_weight_names(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    for name, tensor in weights:
        if name.startswith(EXPLICIT_SKIPS):
            continue
        if name.startswith("audio_tower.model.encoder."):
            name = "audio_tower." + name.removeprefix("audio_tower.model.encoder.")
        yield name, tensor


def validate_projector_loaded(*, expected: set[str], loaded: set[str]) -> None:
    missing = expected - loaded
    if missing:
        raise RuntimeError(f"Partially loaded projector; missing={sorted(missing)}")
