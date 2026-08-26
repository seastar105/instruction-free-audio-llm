import pytest

from audio_lfm_eval.config import BenchmarkSpec
from audio_lfm_eval.data import normalize_row
from audio_lfm_eval.generation import (
    AudioDurationBudget,
    _completed_row_indices,
    _preflight_context,
)
from audio_lfm_eval.http_client import GenerationSettings


def _spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        name="mmau-pro",
        dataset_id="example/data",
        revision="a" * 40,
        subsets=("default",),
        split="test",
        question_mode="text_question",
        official_format="mmau_pro_parquet",
        audio_patterns=("test.parquet",),
        scorer_repo="https://example.invalid",
        scorer_revision="b" * 40,
        supports_multiple_audio=True,
    )


def test_context_preflight_rejects_worst_case_at_16k(tmp_path, monkeypatch) -> None:
    sample = normalize_row(
        {
            "id": "three-long-clips",
            "audio_path": ["first.wav", "second.wav", "third.wav"],
        }
    )
    monkeypatch.setattr(
        "audio_lfm_eval.generation.audio_num_samples",
        lambda value, root: 720 * 16_000,
    )
    with pytest.raises(ValueError, match="cannot fit the full sample") as error:
        _preflight_context(
            [sample],
            data_root=tmp_path,
            spec=_spec(),
            settings=GenerationSettings(max_tokens=1024),
            max_audio_seconds=720,
            max_model_len=16_384,
            audio_chunk_seconds=30,
            audio_stack_factor=4,
            text_prompt_token_reserve=1024,
        )
    assert "[720.000s, 720.000s, 720.000s]" in str(error.value)
    assert "will not be cropped or silently skipped" in str(error.value)


def test_context_preflight_accepts_complete_ten_minute_audio(
    tmp_path, monkeypatch
) -> None:
    sample = normalize_row({"id": "ten-minutes", "audio_path": "long.wav"})
    monkeypatch.setattr(
        "audio_lfm_eval.generation.audio_num_samples",
        lambda value, root: 600 * 16_000,
    )
    _preflight_context(
        [sample],
        data_root=tmp_path,
        spec=_spec(),
        settings=GenerationSettings(max_tokens=1024),
        max_audio_seconds=720,
        max_model_len=32_768,
        audio_chunk_seconds=30,
        audio_stack_factor=4,
        text_prompt_token_reserve=1024,
    )


def test_context_preflight_accepts_three_worst_case_items_at_32k(
    tmp_path, monkeypatch
) -> None:
    sample = normalize_row(
        {
            "id": "three-ten-minute-clips",
            "audio_path": ["first.wav", "second.wav", "third.wav"],
        }
    )
    monkeypatch.setattr(
        "audio_lfm_eval.generation.audio_num_samples",
        lambda value, root: 720 * 16_000,
    )
    _preflight_context(
        [sample],
        data_root=tmp_path,
        spec=_spec(),
        settings=GenerationSettings(max_tokens=1024),
        max_audio_seconds=720,
        max_model_len=32_768,
        audio_chunk_seconds=30,
        audio_stack_factor=4,
        text_prompt_token_reserve=1024,
    )


def test_audio_duration_budget_is_weighted_and_released() -> None:
    budget = AudioDurationBudget(100.0)
    assert budget.available_seconds == 100.0
    with budget.reserve(30.0):
        assert budget.available_seconds == 70.0
    assert budget.available_seconds == 100.0


def test_audio_duration_budget_rejects_one_oversized_request() -> None:
    budget = AudioDurationBudget(100.0)
    with (
        pytest.raises(ValueError, match="host-memory budget"),
        budget.reserve(100.1),
    ):
        pass


def test_resume_uses_ordered_rows_instead_of_duplicate_source_ids() -> None:
    records = [
        {"id": "duplicate", "prediction": "first"},
        {"id": "duplicate", "prediction": "second"},
    ]
    assert _completed_row_indices(records, total_rows=3) == {0, 1}


def test_resume_prefers_explicit_row_indices_and_retries_errors() -> None:
    records = [
        {"id": "duplicate", "row_index": 2, "prediction": "done"},
        {"id": "duplicate", "row_index": 1, "error": "retry me"},
    ]
    assert _completed_row_indices(records, total_rows=3) == {2}
