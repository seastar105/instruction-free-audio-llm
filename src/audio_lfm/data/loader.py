from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset

from audio_lfm.data.mixed_backend import MixedCaptionStewBackend
from audio_lfm.data.types import PrecomputedAudioExample
from audio_lfm.data.worker_preprocessing import (
    AudioPreprocessingConfig,
    WhisperWorkerPreprocessor,
)

Item = TypeVar("Item")


class EpochStreamingDataset(IterableDataset[PrecomputedAudioExample]):
    """Expose one deterministic, worker-sharded backend epoch to DataLoader."""

    def __init__(
        self,
        backend: MixedCaptionStewBackend,
        *,
        epoch: int,
        preprocessing: AudioPreprocessingConfig,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.epoch = epoch
        self.preprocessing = preprocessing

    def __iter__(self) -> Iterator[PrecomputedAudioExample]:
        preprocessor = WhisperWorkerPreprocessor(self.preprocessing)
        for raw in self.backend.iter_epoch(self.epoch):
            yield preprocessor(raw)


def build_epoch_dataloader(
    backend: MixedCaptionStewBackend,
    *,
    epoch: int,
    num_workers: int,
    persistent_workers: bool,
    prefetch_factor: int,
    preprocessing: AudioPreprocessingConfig,
) -> Iterable[PrecomputedAudioExample]:
    """Build an unbatched loader; sequence packing remains in the main process."""
    dataset = EpochStreamingDataset(backend, epoch=epoch, preprocessing=preprocessing)
    if num_workers:
        # The model initializes CUDA before the epoch iterator is consumed.
        # Never fork a CUDA-initialized process. Spawned workers only decode CPU audio.
        loader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            multiprocessing_context="spawn",
        )
    else:
        loader = DataLoader(dataset, batch_size=None, num_workers=0)
    return iter_rank_synchronized(loader)


def iter_rank_synchronized(source: Iterable[Item]) -> Iterator[Item]:
    """End the epoch on every rank as soon as any one rank is exhausted.

    Every rank participates in exactly one exhaustion collective before each yield.
    A rank that fetched an unmatched item discards it instead of entering a model
    collective that an exhausted peer cannot reach.
    """
    iterator = iter(source)
    distributed = (
        dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    )
    if not distributed:
        yield from iterator
        return
    backend = str(dist.get_backend()).lower()
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if "nccl" in backend
        else torch.device("cpu")
    )
    while True:
        try:
            item = next(iterator)
            local_has_data = 1
        except StopIteration:
            item = None
            local_has_data = 0
        all_ranks_have_data = torch.tensor(
            local_has_data, dtype=torch.int32, device=device
        )
        dist.all_reduce(all_ranks_have_data, op=dist.ReduceOp.MIN)
        if not bool(all_ranks_have_data.item()):
            return
        if item is None:
            raise RuntimeError("Exhaustion collective returned an inconsistent result")
        yield item
