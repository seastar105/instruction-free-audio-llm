from __future__ import annotations

import math
import multiprocessing as mp
from collections.abc import Iterable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from torch.utils.data import DataLoader, IterableDataset

from audio_lfm.data.loader import iter_rank_synchronized
from audio_lfm.data.mixed_backend import MixedCaptionStewBackend
from audio_lfm.data.pack_planner import OnlinePackPlanner
from audio_lfm.data.types import (
    DeferredAudioExample,
    DeferredPreparedExample,
    PackedHostItem,
    PreparedExample,
)
from audio_lfm.data.worker_preprocessing import (
    AudioPreprocessingConfig,
    WhisperWorkerPreprocessor,
)
from audio_lfm.model.packed_batch import build_host_audio_batch
from audio_lfm.model.prompt_compiler import PromptCompiler


@dataclass(frozen=True)
class WorkerPackingConfig:
    max_lfm_tokens: int
    max_sample_lfm_tokens: int
    planning_buffer_examples: int
    max_examples_per_pack: int | None
    oversized_example_policy: str
    best_fit_decreasing: bool
    stack_factor: int
    vocabulary_size: int


def _planned_encoder_frames(
    num_samples: int, *, chunk_samples: int, hop_length: int = 160
) -> int:
    if num_samples <= 0:
        raise ValueError("Metadata num_samples must be positive")
    full_chunks, remainder = divmod(num_samples, chunk_samples)
    full_mel_frames = (chunk_samples + hop_length - 1) // hop_length
    full_encoder_frames = (full_mel_frames + 1) // 2
    remainder_encoder_frames = 0
    if remainder:
        remainder_mel_frames = (remainder + hop_length - 1) // hop_length
        remainder_encoder_frames = (remainder_mel_frames + 1) // 2
    return full_chunks * full_encoder_frames + remainder_encoder_frames


class PackedEpochStreamingDataset(IterableDataset[PackedHostItem]):
    """Plan, decode, and featurize complete packs inside each WDS worker."""

    def __init__(
        self,
        backend: MixedCaptionStewBackend,
        *,
        epoch: int,
        preprocessing: AudioPreprocessingConfig,
        prompt_compiler: PromptCompiler,
        packing: WorkerPackingConfig,
        committed_audio_ids: frozenset[str] = frozenset(),
        stop_event: Any | None = None,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.epoch = epoch
        self.preprocessing = preprocessing
        self.prompt_compiler = prompt_compiler
        self.packing = packing
        self.committed_audio_ids = committed_audio_ids
        self.stop_event = stop_event

    def _stopping(self) -> bool:
        return self.stop_event is not None and bool(self.stop_event.is_set())

    def _prepare_deferred(self, raw: DeferredAudioExample) -> DeferredPreparedExample:
        if raw.planned_num_samples <= 0:
            raise ValueError(
                f"Duration sidecar has no num_samples for {raw.audio_id!r}"
            )
        text = self.prompt_compiler.compile(raw.selected_target)
        encoder_frames = _planned_encoder_frames(
            raw.planned_num_samples,
            chunk_samples=self.preprocessing.block_samples,
        )
        projected_frames = math.ceil(encoder_frames / self.packing.stack_factor)
        total = (
            len(text.before_audio_ids)
            + 1
            + projected_frames
            + 1
            + len(text.after_audio_prompt_ids)
            + len(text.target_suffix_ids)
        )
        return DeferredPreparedExample(
            raw=raw,
            text=text,
            estimated_audio_embedding_length=projected_frames,
            estimated_total_lfm_length=total,
        )

    def __iter__(self) -> Iterator[PackedHostItem]:
        preprocessor = WhisperWorkerPreprocessor(self.preprocessing)
        planner = OnlinePackPlanner[DeferredPreparedExample](
            max_lfm_tokens=self.packing.max_lfm_tokens,
            max_sample_lfm_tokens=self.packing.max_sample_lfm_tokens,
            planning_buffer_examples=self.packing.planning_buffer_examples,
            max_examples_per_pack=self.packing.max_examples_per_pack,
            oversized_example_policy=self.packing.oversized_example_policy,
            best_fit_decreasing=self.packing.best_fit_decreasing,
        )
        source = iter(self.backend.iter_deferred_epoch(self.epoch))
        pending_oversized = 0
        pending_decode_failures = 0
        while True:
            if self._stopping():
                return
            planning_buffer: list[DeferredPreparedExample] = []
            previous_planning_failures = self.backend.decode_failure_count
            for _ in range(self.packing.planning_buffer_examples):
                if self._stopping():
                    return
                try:
                    planning_buffer.append(self._prepare_deferred(next(source)))
                except StopIteration:
                    break
            pending_decode_failures += (
                self.backend.decode_failure_count - previous_planning_failures
            )
            if not planning_buffer:
                return
            previous_oversized = planner.oversized_example_count
            plans = planner.pack_buffer(planning_buffer)
            pending_oversized += planner.oversized_example_count - previous_oversized
            for plan in plans:
                if self._stopping():
                    return
                plan_ids = {example.audio_id for example in plan.examples}
                committed = plan_ids.intersection(self.committed_audio_ids)
                if committed:
                    if committed != plan_ids:
                        raise RuntimeError(
                            "Resume state intersects only part of a deterministic "
                            f"worker pack: {sorted(committed)}"
                        )
                    continue
                realized: list[PreparedExample] = []
                previous_decode_failures = self.backend.decode_failure_count
                for deferred in plan.examples:
                    if self._stopping():
                        return
                    raw = self.backend.decode_deferred(deferred.raw, epoch=self.epoch)
                    if raw is None:
                        continue
                    precomputed = preprocessor(raw)
                    if self._stopping():
                        return
                    actual_encoder_frames = sum(precomputed.effective_encoder_lengths)
                    actual_projected_frames = math.ceil(
                        actual_encoder_frames / self.packing.stack_factor
                    )
                    actual_total = (
                        len(deferred.text.before_audio_ids)
                        + 1
                        + actual_projected_frames
                        + 1
                        + len(deferred.text.after_audio_prompt_ids)
                        + len(deferred.text.target_suffix_ids)
                    )
                    if (
                        actual_projected_frames
                        != deferred.estimated_audio_embedding_length
                        or actual_total != deferred.estimated_total_lfm_length
                    ):
                        raise RuntimeError(
                            "Metadata/audio length mismatch for "
                            f"{deferred.audio_id!r}: "
                            f"planned={deferred.estimated_total_lfm_length}, "
                            f"actual={actual_total}"
                        )
                    realized.append(
                        PreparedExample(
                            raw=precomputed,
                            text=deferred.text,
                            estimated_audio_embedding_length=actual_projected_frames,
                            estimated_total_lfm_length=actual_total,
                        )
                    )
                pending_decode_failures += (
                    self.backend.decode_failure_count - previous_decode_failures
                )
                if not realized:
                    continue
                yield PackedHostItem(
                    batch=build_host_audio_batch(
                        examples=realized,
                        vocabulary_size=self.packing.vocabulary_size,
                        max_lfm_tokens=self.packing.max_lfm_tokens,
                    ),
                    oversized_examples_skipped=pending_oversized,
                    decode_failures_skipped=pending_decode_failures,
                )
                pending_oversized = 0
                pending_decode_failures = 0


def _identity(value: Any) -> Any:
    return value


def _filter_committed_packs(
    source: Iterable[PackedHostItem], committed_audio_ids: frozenset[str]
) -> Iterator[PackedHostItem]:
    seen: set[str] = set()
    for item in source:
        audio_ids = item.batch.layout.audio_ids
        unique = set(audio_ids)
        if len(unique) != len(audio_ids):
            raise ValueError("A packed batch contains duplicate audio IDs")
        repeated = seen.intersection(unique)
        if repeated:
            raise ValueError(f"Duplicate streamed audio IDs: {sorted(repeated)}")
        seen.update(unique)
        committed = unique.intersection(committed_audio_ids)
        if committed:
            if committed != unique:
                raise RuntimeError(
                    "Resume state intersects only part of a deterministic pack: "
                    f"{sorted(committed)}"
                )
            continue
        yield item


IteratorT = TypeVar("IteratorT")


class _OwnedIterator(Iterator[IteratorT], Generic[IteratorT]):
    """Own a DataLoader iterator and shut its workers down on early close."""

    def __init__(
        self,
        source: Iterator[IteratorT],
        loader_iterator: Any,
        stop_event: Any | None,
    ) -> None:
        self.source = source
        self.loader_iterator = loader_iterator
        self.stop_event = stop_event
        self.closed = False

    def __iter__(self) -> _OwnedIterator[IteratorT]:
        return self

    def __next__(self) -> IteratorT:
        if self.closed:
            raise StopIteration
        try:
            return next(self.source)
        except StopIteration:
            self._close(drain=False)
            raise

    def close(self) -> None:
        self._close(drain=True)

    def _close(self, *, drain: bool) -> None:
        if self.closed:
            return
        self.closed = True
        if self.stop_event is not None:
            self.stop_event.set()
        try:
            close = getattr(self.source, "close", None)
            if callable(close):
                close()
            if drain and self.stop_event is not None:
                try:
                    while True:
                        next(self.loader_iterator)
                except Exception:
                    # Closing early discards prefetched work. A prefetched worker
                    # failure must not prevent the owner from joining every worker.
                    pass
        finally:
            shutdown = getattr(self.loader_iterator, "_shutdown_workers", None)
            if callable(shutdown):
                shutdown()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


def build_packed_epoch_dataloader(
    backend: MixedCaptionStewBackend,
    *,
    epoch: int,
    committed_audio_ids: frozenset[str],
    num_workers: int,
    persistent_workers: bool,
    prefetch_factor: int,
    preprocessing: AudioPreprocessingConfig,
    prompt_compiler: PromptCompiler,
    packing: WorkerPackingConfig,
) -> Iterable[PackedHostItem]:
    context = mp.get_context("spawn") if num_workers else None
    stop_event = context.Event() if context is not None else None
    dataset = PackedEpochStreamingDataset(
        backend,
        epoch=epoch,
        preprocessing=preprocessing,
        prompt_compiler=prompt_compiler,
        packing=packing,
        committed_audio_ids=committed_audio_ids,
        stop_event=stop_event,
    )
    if num_workers:
        loader = DataLoader(
            dataset,
            batch_size=None,
            collate_fn=_identity,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            multiprocessing_context=context,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=None,
            collate_fn=_identity,
            num_workers=0,
        )
    loader_iterator = iter(loader)
    uncommitted = _filter_committed_packs(loader_iterator, committed_audio_ids)
    synchronized = iter(iter_rank_synchronized(uncommitted))
    return _OwnedIterator(synchronized, loader_iterator, stop_event)
