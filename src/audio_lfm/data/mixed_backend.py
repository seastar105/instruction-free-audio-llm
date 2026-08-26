from __future__ import annotations

import random
from collections.abc import Iterator, Sequence

from audio_lfm.data.captionstew_backend import CaptionStewBackend
from audio_lfm.data.types import DeferredAudioExample, RawAudioExample


class MixedCaptionStewBackend:
    """Consume every configured dataset once per epoch in deterministic order."""

    def __init__(self, backends: Sequence[CaptionStewBackend], *, seed: int) -> None:
        if not backends:
            raise ValueError("Mixed training requires at least one backend")
        self.backends = tuple(backends)
        self.seed = seed

    def iter_epoch(self, epoch: int) -> Iterator[RawAudioExample]:
        rng = random.Random(self.seed + epoch)
        active = [iter(backend.iter_epoch(epoch)) for backend in self.backends]
        while active:
            index = rng.randrange(len(active))
            try:
                yield next(active[index])
            except StopIteration:
                active.pop(index)

    def iter_deferred_epoch(self, epoch: int) -> Iterator[DeferredAudioExample]:
        rng = random.Random(self.seed + epoch)
        active = [iter(backend.iter_deferred_epoch(epoch)) for backend in self.backends]
        while active:
            index = rng.randrange(len(active))
            try:
                yield next(active[index])
            except StopIteration:
                active.pop(index)

    def decode_deferred(
        self, deferred: DeferredAudioExample, *, epoch: int
    ) -> RawAudioExample | None:
        for backend in self.backends:
            if backend.dataset == deferred.catalog_record.dataset:
                return backend.decode_deferred(deferred, epoch=epoch)
        raise KeyError(
            f"No backend configured for dataset {deferred.catalog_record.dataset!r}"
        )

    @property
    def decode_failure_count(self) -> int:
        return sum(backend.decode_failure_count for backend in self.backends)
