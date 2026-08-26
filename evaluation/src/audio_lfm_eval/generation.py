from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from audio_lfm_eval.audio import (
    TARGET_SAMPLE_RATE,
    audio_file_uri,
    audio_num_samples,
    projected_audio_tokens,
)
from audio_lfm_eval.config import BenchmarkSpec
from audio_lfm_eval.data import EvalSample, iter_samples
from audio_lfm_eval.http_client import GenerationSettings, VllmHttpClient
from audio_lfm_eval.predictions import PredictionStore, write_manifest


class AudioDurationBudget:
    """Bound decoded/base64 audio retained by concurrent HTTP requests."""

    def __init__(self, capacity_seconds: float) -> None:
        if capacity_seconds <= 0:
            raise ValueError("capacity_seconds must be positive")
        self.capacity_seconds = float(capacity_seconds)
        self._available_seconds = float(capacity_seconds)
        self._condition = threading.Condition()

    @property
    def available_seconds(self) -> float:
        with self._condition:
            return self._available_seconds

    @contextmanager
    def reserve(self, seconds: float) -> Iterator[None]:
        amount = max(float(seconds), 1.0 / TARGET_SAMPLE_RATE)
        if amount > self.capacity_seconds + 1e-9:
            raise ValueError(
                f"one request needs {amount:.3f}s of in-flight audio, exceeding "
                f"the {self.capacity_seconds:.3f}s host-memory budget"
            )
        with self._condition:
            self._condition.wait_for(lambda: self._available_seconds + 1e-9 >= amount)
            self._available_seconds -= amount
        try:
            yield
        finally:
            with self._condition:
                self._available_seconds += amount
                self._condition.notify_all()


def _run_one(
    sample: EvalSample,
    *,
    row_index: int,
    client: VllmHttpClient,
    model: str,
    data_root: Path,
    spec: BenchmarkSpec,
    settings: GenerationSettings,
    max_audio_seconds: float,
    audio_budget: AudioDurationBudget,
) -> dict[str, Any]:
    request_audio_seconds = sum(
        audio_num_samples(value, data_root) / TARGET_SAMPLE_RATE
        for value in sample.audio_values
    )
    with audio_budget.reserve(request_audio_seconds):
        encoded = [
            audio_file_uri(value, data_root, max_seconds=max_audio_seconds)
            for value in sample.audio_values
        ]
        question = sample.question if spec.question_mode == "text_question" else ""
        result = client.generate(
            model=model,
            audio_urls=[item[0] for item in encoded],
            question=question,
            choices=sample.choices if spec.question_mode == "text_question" else (),
            settings=settings,
        )
    return {
        "id": sample.sample_id,
        "row_index": row_index,
        "benchmark": spec.name,
        "prediction": result["prediction"],
        "finish_reason": result["finish_reason"],
        "truncated": result["finish_reason"] == "length",
        "usage": result["usage"],
        "response_id": result["response_id"],
        "audio": [item[1] for item in encoded],
        "source_record": sample.source_record,
    }


def _completed_row_indices(
    records: list[dict[str, Any]], *, total_rows: int
) -> set[int]:
    """Recover ordered legacy rows and explicit indices without trusting IDs."""
    completed = set()
    for legacy_index, record in enumerate(records):
        explicit = record.get("row_index")
        if explicit is None:
            if "error" in record:
                raise RuntimeError(
                    "Cannot safely resume a legacy prediction file containing "
                    "errors because its source IDs may be duplicated"
                )
            row_index = legacy_index
        elif isinstance(explicit, int):
            row_index = explicit
        else:
            raise ValueError(f"Invalid prediction row_index {explicit!r}")
        if row_index < 0 or row_index >= total_rows:
            raise ValueError(
                f"Prediction row_index {row_index} is outside 0..{total_rows - 1}"
            )
        if "error" not in record:
            completed.add(row_index)
    return completed


def _preflight_context(
    samples: list[EvalSample],
    *,
    data_root: Path,
    spec: BenchmarkSpec,
    settings: GenerationSettings,
    max_audio_seconds: float,
    max_model_len: int,
    audio_chunk_seconds: float,
    audio_stack_factor: int,
    text_prompt_token_reserve: int,
) -> None:
    chunk_samples = round(audio_chunk_seconds * TARGET_SAMPLE_RATE)
    for sample in samples:
        if len(sample.audio_values) > 3:
            raise ValueError(
                f"{spec.name} row {sample.sample_id} has "
                f"{len(sample.audio_values)} audio items; the server limit is 3"
            )
        item_samples = [
            audio_num_samples(value, data_root) for value in sample.audio_values
        ]
        too_long = [
            count / TARGET_SAMPLE_RATE
            for count in item_samples
            if count > max_audio_seconds * TARGET_SAMPLE_RATE + 1e-6
        ]
        if too_long:
            raise ValueError(
                f"{spec.name} row {sample.sample_id} contains a "
                f"{max(too_long):.3f}s audio item, exceeding the configured "
                f"{max_audio_seconds:.3f}s per-item evaluation limit"
            )
        audio_tokens = sum(
            projected_audio_tokens(
                count,
                chunk_samples=chunk_samples,
                stack_factor=audio_stack_factor,
            )
            for count in item_samples
        )
        required = audio_tokens + text_prompt_token_reserve + settings.max_tokens
        if required > max_model_len:
            durations = ", ".join(
                f"{count / TARGET_SAMPLE_RATE:.3f}s" for count in item_samples
            )
            raise ValueError(
                f"{spec.name} row {sample.sample_id} cannot fit the full sample: "
                f"audio durations [{durations}] require {audio_tokens} audio "
                f"tokens and at least {required} total context tokens including "
                f"the {text_prompt_token_reserve}-token prompt reserve and "
                f"{settings.max_tokens} output tokens, but max_model_len is "
                f"{max_model_len}. Increase max_model_len; audio will not be "
                "cropped or silently skipped."
            )


def generate_benchmark(
    *,
    client: VllmHttpClient,
    model: str,
    model_identity: dict[str, str],
    spec: BenchmarkSpec,
    subset: str,
    data_root: str | Path,
    output_root: str | Path,
    settings: GenerationSettings,
    concurrency: int,
    max_audio_seconds: float,
    max_model_len: int,
    audio_chunk_seconds: float,
    audio_stack_factor: int,
    text_prompt_token_reserve: int,
    max_inflight_audio_seconds: float,
    limit: int | None,
) -> dict[str, int]:
    if subset not in spec.subsets:
        raise ValueError(f"Unknown {spec.name} subset {subset!r}")
    output_dir = Path(output_root) / spec.name / subset
    store = PredictionStore(output_dir)
    manifest = {
        "format_version": 1,
        "transport": "vllm-openai-http",
        "benchmark": spec.name,
        "subset": subset,
        "dataset_id": spec.dataset_id,
        "dataset_revision": spec.revision,
        "model": model,
        "model_identity": model_identity,
        "generation": settings.to_dict(),
        "max_audio_seconds": max_audio_seconds,
        "max_model_len": max_model_len,
        "audio_chunk_seconds": audio_chunk_seconds,
        "audio_stack_factor": audio_stack_factor,
        "text_prompt_token_reserve": text_prompt_token_reserve,
        "supports_long_audio_chunking": True,
        "max_audio_items_per_prompt": 3,
    }
    write_manifest(output_dir, manifest)
    all_samples = list(iter_samples(data_root, subset, limit=limit))
    _preflight_context(
        all_samples,
        data_root=Path(data_root),
        spec=spec,
        settings=settings,
        max_audio_seconds=max_audio_seconds,
        max_model_len=max_model_len,
        audio_chunk_seconds=audio_chunk_seconds,
        audio_stack_factor=audio_stack_factor,
        text_prompt_token_reserve=text_prompt_token_reserve,
    )
    completed = _completed_row_indices(store.records(), total_rows=len(all_samples))
    indexed_samples = [
        (row_index, item)
        for row_index, item in enumerate(all_samples)
        if row_index not in completed
    ]
    succeeded = 0
    failed = 0
    audio_budget = AudioDurationBudget(max_inflight_audio_seconds)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                _run_one,
                sample,
                row_index=row_index,
                client=client,
                model=model,
                data_root=Path(data_root),
                spec=spec,
                settings=settings,
                max_audio_seconds=max_audio_seconds,
                audio_budget=audio_budget,
            )
            for row_index, sample in indexed_samples
        ]
        for (row_index, sample), future in zip(indexed_samples, futures, strict=True):
            try:
                store.append(future.result())
                succeeded += 1
            except Exception as error:
                store.append(
                    {
                        "id": sample.sample_id,
                        "row_index": row_index,
                        "benchmark": spec.name,
                        "error": f"{type(error).__name__}: {error}",
                        "source_record": sample.source_record,
                    }
                )
                failed += 1
    progress = {"succeeded": succeeded, "failed": failed, "previous": len(completed)}
    (output_dir / "progress.json").write_text(
        json.dumps(progress, indent=2, sort_keys=True), encoding="utf-8"
    )
    return progress
