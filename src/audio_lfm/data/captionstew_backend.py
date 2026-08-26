from __future__ import annotations

import inspect
import os
import random
import warnings
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path
from typing import Any, cast

from audio_lfm.data.catalog import CatalogIndex
from audio_lfm.data.decode import (
    DataContractError,
    DurationSidecar,
    LongAudioSkipped,
    decode_webdataset_sample,
)
from audio_lfm.data.local_shards import complete_local_shards
from audio_lfm.data.targets import select_target, validate_target_consistency
from audio_lfm.data.types import (
    CatalogAudioRecord,
    DeferredAudioExample,
    LocalSampleReference,
    RawAudioExample,
)


class CaptionStewBackend:
    def __init__(
        self,
        *,
        captionstew_root: str | Path,
        dataset: str,
        catalog: CatalogIndex,
        shard_shuffle: int,
        sample_shuffle: int,
        max_audio_seconds: float,
        long_audio_policy: str,
        strict_target_consistency: bool = False,
        max_bad_samples: int = 0,
        seed: int = 1337,
        local_samples: Iterable[dict[str, Any]] | None = None,
        duration_sidecar: DurationSidecar | None = None,
    ) -> None:
        self.captionstew_root = Path(captionstew_root)
        self.dataset = dataset
        self.catalog = catalog
        self.shard_shuffle = shard_shuffle
        self.sample_shuffle = sample_shuffle
        self.max_audio_seconds = max_audio_seconds
        self.long_audio_policy = long_audio_policy
        self.strict_target_consistency = strict_target_consistency
        self.max_bad_samples = max_bad_samples
        self.seed = seed
        self.local_samples = local_samples
        self.duration_sidecar = duration_sidecar
        self.long_audio_skip_count = 0
        self.decode_failure_count = 0

    def _open(self, epoch: int) -> Iterable[dict[str, Any]]:
        if self.local_samples is not None:
            # Production WebDataset streams apply ``split_by_worker`` below.
            # Mirror that behavior for local/test iterables to prevent duplicates.
            from torch.utils.data import get_worker_info

            worker = get_worker_info()
            if worker is None:
                return self.local_samples
            return islice(iter(self.local_samples), worker.id, None, worker.num_workers)
        kwargs: dict[str, Any] = {}
        local_shards = complete_local_shards(self.captionstew_root, self.catalog)
        if local_shards is not None:
            import webdataset as wds

            return cast(
                Iterable[dict[str, Any]],
                wds.WebDataset(
                    list(local_shards),
                    shardshuffle=self.shard_shuffle,
                    nodesplitter=_webdataset_splitter("split_by_node"),
                    workersplitter=_webdataset_splitter("split_by_worker"),
                    seed=self.seed + epoch,
                ).shuffle(self.sample_shuffle, rng=random.Random(self.seed + epoch))
                if self.sample_shuffle
                else wds.WebDataset(
                    list(local_shards),
                    shardshuffle=self.shard_shuffle,
                    nodesplitter=_webdataset_splitter("split_by_node"),
                    workersplitter=_webdataset_splitter("split_by_worker"),
                    seed=self.seed + epoch,
                ),
            )
        try:
            from captionstew.training_client import open_webdataset
        except ImportError as error:
            raise RuntimeError(
                "CaptionStew training client is not installed; install "
                "$CAPTIONSTEW_REPO[training] with uv pip"
            ) from error
        signature = inspect.signature(open_webdataset)
        available = signature.parameters
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in available.values()
        )
        value_by_name: dict[str, Any] = {
            "root": self.captionstew_root,
            "captionstew_root": self.captionstew_root,
            "path": self.captionstew_root,
            "dataset": self.dataset,
            "dataset_name": self.dataset,
            "shard_shuffle": self.shard_shuffle,
            "shardshuffle": self.shard_shuffle,
            "shuffle": self.shard_shuffle,
            "nodesplitter": _webdataset_splitter("split_by_node"),
            "workersplitter": _webdataset_splitter("split_by_worker"),
            "seed": self.seed + epoch,
        }
        for name, value in value_by_name.items():
            forwarded_splitter = accepts_kwargs and name in {
                "nodesplitter",
                "seed",
                "workersplitter",
            }
            if (name in available or forwarded_splitter) and value is not None:
                kwargs[name] = value
        required_positional = [
            parameter
            for parameter in available.values()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            and parameter.name not in kwargs
        ]
        args: list[Any] = []
        for parameter in required_positional:
            if parameter.name in {"url", "urls", "bucket_root"}:
                args.append(str(self.captionstew_root))
            else:
                raise RuntimeError(
                    f"Unsupported required open_webdataset parameter: {parameter.name}"
                )
        stream = open_webdataset(*args, **kwargs)
        if self.sample_shuffle and hasattr(stream, "shuffle"):
            stream = stream.shuffle(
                self.sample_shuffle, rng=random.Random(self.seed + epoch)
            )
        return cast(Iterable[dict[str, Any]], stream)

    def _open_local_references(
        self, epoch: int
    ) -> Iterator[tuple[CatalogAudioRecord, LocalSampleReference]] | None:
        if self.local_samples is not None or self.duration_sidecar is None:
            return None
        if complete_local_shards(self.captionstew_root, self.catalog) is None:
            return None
        references: list[tuple[CatalogAudioRecord, LocalSampleReference]] = []
        for record in self.catalog.audio_by_id.values():
            reference = self.duration_sidecar.local_reference(record)
            if reference is None:
                return None
            references.append((record, reference))
        references.sort(key=lambda item: (item[0].wds_shard, item[0].wds_key))
        shards = list(self.catalog.selected_shards)
        if self.shard_shuffle:
            import webdataset as wds

            shards = list(
                wds.shuffle(
                    bufsize=self.shard_shuffle,
                    rng=random.Random(self.seed + epoch),
                )(iter(shards))
            )
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        shards = shards[rank::world_size]
        from torch.utils.data import get_worker_info

        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        selected_shards = set(shards[worker_id::worker_count])
        selected = (item for item in references if item[0].wds_shard in selected_shards)
        if not self.sample_shuffle:
            return iter(selected)
        import webdataset as wds

        return cast(
            Iterator[tuple[CatalogAudioRecord, LocalSampleReference]],
            iter(
                wds.shuffle(
                    bufsize=self.sample_shuffle,
                    rng=random.Random(self.seed + epoch),
                )(selected)
            ),
        )

    def _load_local_sample(self, reference: LocalSampleReference) -> dict[str, Any]:
        path = self.captionstew_root / reference.wds_shard

        with path.open("rb") as stream:

            def read_range(offset: int, size: int) -> bytes:
                stream.seek(offset)
                value = stream.read(size)
                if len(value) != size:
                    raise DataContractError(
                        f"Short TAR read in {reference.wds_shard!r} at {offset}: "
                        f"{len(value)} != {size}"
                    )
                return value

            flac = read_range(reference.flac_offset, reference.flac_size)
            metadata = read_range(reference.json_offset, reference.json_size)

        return {
            "__key__": reference.wds_key,
            "flac": flac,
            "json": metadata,
        }

    def _skip_invalid_planned_audio(self, audio_id: str, reason: str) -> None:
        self.decode_failure_count += 1
        if self.decode_failure_count > self.max_bad_samples:
            raise DataContractError(
                f"{self.dataset} audio {audio_id!r} is invalid: {reason}"
            )
        warnings.warn(
            f"Skipping invalid {self.dataset} audio {audio_id!r}: {reason}",
            stacklevel=2,
        )

    def iter_deferred_epoch(self, epoch: int) -> Iterator[DeferredAudioExample]:
        """Yield WDS-shuffled samples before FLAC decode or log-Mel extraction."""
        local_references = self._open_local_references(epoch)
        if local_references is not None:
            if self.duration_sidecar is None:
                raise RuntimeError("Local references require a duration sidecar")
            for catalog_record, reference in local_references:
                audio_id = catalog_record.audio_id
                styles = self.catalog.style_captions_by_id[audio_id]
                transcript = self.catalog.transcript_by_id[audio_id]
                duration = self.duration_sidecar.get(catalog_record)
                if duration is None:
                    raise DataContractError(f"Duration sidecar lacks {audio_id!r}")
                if duration[0] <= 0:
                    self._skip_invalid_planned_audio(
                        audio_id, "exact metadata identifies an empty FLAC"
                    )
                    continue
                yield DeferredAudioExample(
                    audio_id=audio_id,
                    sample=None,
                    local_reference=reference,
                    catalog_record=catalog_record,
                    style_captions=styles,
                    transcript=transcript,
                    selected_target=select_target(
                        styles, seed=self.seed, epoch=epoch, audio_id=audio_id
                    ),
                    planned_num_samples=duration[0],
                )
            return
        for sample in self._open(epoch):
            audio_id = str(sample.get("__key__", ""))
            if audio_id not in self.catalog.allowed_audio_ids:
                continue
            catalog_record = self.catalog.audio_by_id[audio_id]
            planned_num_samples = 0
            if self.duration_sidecar is not None:
                duration = self.duration_sidecar.get(catalog_record)
                if duration is not None:
                    planned_num_samples = duration[0]
                    if planned_num_samples <= 0:
                        self._skip_invalid_planned_audio(
                            audio_id, "exact metadata identifies an empty FLAC"
                        )
                        continue
            styles = self.catalog.style_captions_by_id[audio_id]
            transcript = self.catalog.transcript_by_id[audio_id]
            yield DeferredAudioExample(
                audio_id=audio_id,
                sample=sample,
                local_reference=None,
                catalog_record=catalog_record,
                style_captions=styles,
                transcript=transcript,
                selected_target=select_target(
                    styles, seed=self.seed, epoch=epoch, audio_id=audio_id
                ),
                planned_num_samples=planned_num_samples,
            )

    def decode_deferred(
        self, deferred: DeferredAudioExample, *, epoch: int
    ) -> RawAudioExample | None:
        catalog_record = deferred.catalog_record
        audio_id = deferred.audio_id
        sample = deferred.sample
        if deferred.local_reference is not None:
            sample = self._load_local_sample(deferred.local_reference)
        if sample is None:
            raise DataContractError(f"No audio payload is available for {audio_id!r}")
        try:
            waveform, rate, metadata, crop_start, original = decode_webdataset_sample(
                sample,
                catalog_record=catalog_record,
                max_audio_seconds=self.max_audio_seconds,
                long_audio_policy=self.long_audio_policy,
                seed=self.seed,
                epoch=epoch,
            )
        except LongAudioSkipped:
            self.long_audio_skip_count += 1
            return None
        except DataContractError as error:
            self.decode_failure_count += 1
            if self.decode_failure_count > self.max_bad_samples:
                raise DataContractError(
                    f"{self.dataset} audio {audio_id!r} failed validation"
                ) from error
            warnings.warn(
                f"Skipping invalid {self.dataset} audio {audio_id!r}: {error}",
                stacklevel=2,
            )
            return None
        if deferred.planned_num_samples and original != deferred.planned_num_samples:
            raise DataContractError(
                f"Duration sidecar mismatch for {audio_id!r}: planned "
                f"{deferred.planned_num_samples}, decoded {original} samples"
            )
        if self.strict_target_consistency:
            validate_target_consistency(
                metadata,
                [
                    *deferred.style_captions,
                    *([deferred.transcript] if deferred.transcript is not None else []),
                ],
            )
        return RawAudioExample(
            audio_id=audio_id,
            waveform=waveform,
            sample_rate=rate,
            source_id=catalog_record.source_id,
            splits=catalog_record.splits,
            style_captions=deferred.style_captions,
            transcript=deferred.transcript,
            selected_target=deferred.selected_target,
            metadata=metadata,
            crop_start_sample=crop_start,
            original_num_samples=original,
        )

    def iter_epoch(self, epoch: int) -> Iterator[RawAudioExample]:
        for deferred in self.iter_deferred_epoch(epoch):
            raw = self.decode_deferred(deferred, epoch=epoch)
            if raw is not None:
                yield raw

    def __iter__(self) -> Iterator[RawAudioExample]:
        return self.iter_epoch(0)


def _webdataset_splitter(name: str) -> Any | None:
    try:
        import webdataset as wds
    except ImportError:
        return None
    return getattr(wds, name, None)
