from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from audio_lfm.data.captionstew_backend import CaptionStewBackend
from audio_lfm.data.catalog import CatalogIndex
from audio_lfm.data.decode import DurationSidecar
from audio_lfm.data.local_shards import LOCAL_MIRROR_MARKER
from audio_lfm.data.mixed_backend import MixedCaptionStewBackend
from audio_lfm.data.packed_loader import (
    PackedEpochStreamingDataset,
    WorkerPackingConfig,
    build_packed_epoch_dataloader,
)
from audio_lfm.data.types import PreparedText, TargetRecord
from audio_lfm.data.worker_preprocessing import AudioPreprocessingConfig
from scripts.build_exact_duration_sidecar import _exact_samples


class StubPromptCompiler:
    def compile(self, target: TargetRecord) -> PreparedText:
        return PreparedText(
            before_audio_ids=(1, 2),
            after_audio_prompt_ids=(3,),
            target_suffix_ids=(4, 5),
            target_id=target.target_id,
            prompt_sha256="test-prompt",
        )


def _packed_fixture(
    tiny_captionstew: dict[str, object], tmp_path: Path
) -> tuple[MixedCaptionStewBackend, WorkerPackingConfig]:
    samples = tiny_captionstew["samples"]
    assert isinstance(samples, list)
    rows: list[dict[str, Any]] = []
    for sample in samples:
        assert isinstance(sample, dict)
        flac = sample["flac"]
        assert isinstance(flac, bytes)
        with sf.SoundFile(io.BytesIO(flac)) as audio:
            frames = int(audio.frames)
        rows.append(
            {
                "audio_id": str(sample["__key__"]),
                "num_samples": frames,
                "duration_seconds": frames / 16_000,
                "flac_sha256": hashlib.sha256(flac).hexdigest(),
            }
        )
    sidecar_path = tmp_path / "durations.parquet"
    pq.write_table(pa.Table.from_pylist(rows), sidecar_path)
    catalog = CatalogIndex.load(
        root=tiny_captionstew["root"],
        dataset="ParaSpeechCaps-Base",
        logical_split="train_base",
    )
    backend = CaptionStewBackend(
        captionstew_root=tiny_captionstew["root"],
        dataset="ParaSpeechCaps-Base",
        catalog=catalog,
        shard_shuffle=0,
        sample_shuffle=0,
        max_audio_seconds=30,
        long_audio_policy="chunk_pad",
        local_samples=samples,
        duration_sidecar=DurationSidecar(sidecar_path),
    )
    mixed = MixedCaptionStewBackend((backend,), seed=1337)
    packing = WorkerPackingConfig(
        max_lfm_tokens=512,
        max_sample_lfm_tokens=512,
        planning_buffer_examples=2,
        max_examples_per_pack=None,
        oversized_example_policy="skip",
        best_fit_decreasing=True,
        stack_factor=4,
        vocabulary_size=512,
    )
    return mixed, packing


def test_worker_plans_then_yields_complete_host_batch(
    tiny_captionstew: dict[str, object], tmp_path: Path
) -> None:
    backend, packing = _packed_fixture(tiny_captionstew, tmp_path)
    dataset = PackedEpochStreamingDataset(
        backend,
        epoch=0,
        preprocessing=AudioPreprocessingConfig(model_id=None, revision=None),
        prompt_compiler=StubPromptCompiler(),  # type: ignore[arg-type]
        packing=packing,
    )
    items = list(dataset)
    assert len(items) == 1
    item = items[0]
    assert item.batch.layout.audio_ids == ["train-0", "train-1"]
    assert item.batch.layout.input_token_count <= packing.max_lfm_tokens
    assert item.batch.input_features.shape == (2, 80, 3000)
    assert item.batch.encoder_frame_mask.shape == (2, 1500)
    assert item.batch.audio_seconds > 0


def test_oversized_examples_are_rejected_before_audio_decode(
    tiny_captionstew: dict[str, object], tmp_path: Path, monkeypatch
) -> None:
    backend, packing = _packed_fixture(tiny_captionstew, tmp_path)
    packing = WorkerPackingConfig(
        **{**packing.__dict__, "max_lfm_tokens": 2, "max_sample_lfm_tokens": 2}
    )

    def fail_decode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("oversized audio must not be decoded")

    monkeypatch.setattr(backend, "decode_deferred", fail_decode)
    dataset = PackedEpochStreamingDataset(
        backend,
        epoch=0,
        preprocessing=AudioPreprocessingConfig(model_id=None, revision=None),
        prompt_compiler=StubPromptCompiler(),  # type: ignore[arg-type]
        packing=packing,
    )
    assert list(dataset) == []


def test_local_planner_does_not_read_flac_before_pack_selection(
    tiny_captionstew: dict[str, object], tmp_path: Path, monkeypatch
) -> None:
    root = tiny_captionstew["root"]
    assert isinstance(root, Path)
    catalog = CatalogIndex.load(
        root=root,
        dataset="ParaSpeechCaps-Base",
        logical_split="train_base",
    )
    exact = _exact_samples(root, catalog.dataset, set(catalog.allowed_audio_ids))
    sidecar_path = tmp_path / "exact-durations.parquet"
    rows = [
        {
            "audio_id": audio_id,
            "duration_seconds": int(reference["num_samples"]) / 16_000,
            "flac_sha256": catalog.audio_by_id[audio_id].flac_sha256,
            "num_samples_source": "flac_streaminfo",
            **reference,
        }
        for audio_id, reference in exact.items()
    ]
    pq.write_table(pa.Table.from_pylist(rows), sidecar_path)
    shard = root / catalog.selected_shards[0]
    marker = {
        "format_version": 1,
        "dataset": catalog.dataset,
        "files": {catalog.selected_shards[0]: shard.stat().st_size},
    }
    (shard.parent / LOCAL_MIRROR_MARKER).write_text(json.dumps(marker))
    backend = CaptionStewBackend(
        captionstew_root=root,
        dataset=catalog.dataset,
        catalog=catalog,
        shard_shuffle=0,
        sample_shuffle=0,
        max_audio_seconds=30,
        long_audio_policy="chunk_pad",
        duration_sidecar=DurationSidecar(sidecar_path, require_exact=True),
    )
    mixed = MixedCaptionStewBackend((backend,), seed=1337)
    packing = WorkerPackingConfig(
        max_lfm_tokens=2,
        max_sample_lfm_tokens=2,
        planning_buffer_examples=2,
        max_examples_per_pack=None,
        oversized_example_policy="skip",
        best_fit_decreasing=True,
        stack_factor=4,
        vocabulary_size=512,
    )

    def fail_read(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("oversized planning must not read a TAR payload")

    monkeypatch.setattr(backend, "_load_local_sample", fail_read)
    dataset = PackedEpochStreamingDataset(
        mixed,
        epoch=0,
        preprocessing=AudioPreprocessingConfig(model_id=None, revision=None),
        prompt_compiler=StubPromptCompiler(),  # type: ignore[arg-type]
        packing=packing,
    )
    assert list(dataset) == []


def test_resume_skips_only_complete_deterministic_packs(
    tiny_captionstew: dict[str, object], tmp_path: Path
) -> None:
    backend, packing = _packed_fixture(tiny_captionstew, tmp_path)
    common = {
        "backend": backend,
        "epoch": 0,
        "num_workers": 0,
        "persistent_workers": False,
        "prefetch_factor": 1,
        "preprocessing": AudioPreprocessingConfig(model_id=None, revision=None),
        "prompt_compiler": StubPromptCompiler(),
        "packing": packing,
    }
    assert (
        list(
            build_packed_epoch_dataloader(
                committed_audio_ids=frozenset({"train-0", "train-1"}),
                **common,  # type: ignore[arg-type]
            )
        )
        == []
    )
    with pytest.raises(RuntimeError, match="intersects only part"):
        list(
            build_packed_epoch_dataloader(
                committed_audio_ids=frozenset({"train-0"}),
                **common,  # type: ignore[arg-type]
            )
        )


def test_multiprocess_workers_handoff_complete_packs(
    tiny_captionstew: dict[str, object], tmp_path: Path
) -> None:
    if os.environ.get("AUDIO_LFM_TEST_MULTIPROCESSING") != "1":
        pytest.skip("requires multiprocessing sockets outside the file sandbox")
    backend, packing = _packed_fixture(tiny_captionstew, tmp_path)
    items = list(
        build_packed_epoch_dataloader(
            backend,
            epoch=0,
            committed_audio_ids=frozenset(),
            num_workers=2,
            persistent_workers=False,
            prefetch_factor=2,
            preprocessing=AudioPreprocessingConfig(model_id=None, revision=None),
            prompt_compiler=StubPromptCompiler(),  # type: ignore[arg-type]
            packing=packing,
        )
    )
    assert sorted(
        audio_id for item in items for audio_id in item.batch.layout.audio_ids
    ) == ["train-0", "train-1"]
    assert all(item.batch.input_features.shape[0] == 1 for item in items)
