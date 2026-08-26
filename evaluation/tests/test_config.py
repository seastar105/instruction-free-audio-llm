from pathlib import Path

from audio_lfm_eval.cli import _runtime
from audio_lfm_eval.config import load_specs


def test_all_requested_benchmarks_are_pinned() -> None:
    specs = load_specs(Path(__file__).parents[1] / "benchmarks.yaml")
    assert set(specs) == {
        "voicebench",
        "mmau",
        "mmsu",
        "mmau-pro",
        "mmar",
        "kvoicebench",
        "kmmau",
        "voicebench-ja",
    }
    for spec in specs.values():
        assert len(spec.revision) == 40
        assert len(spec.scorer_revision) == 40
        assert spec.subsets
        assert set(spec.judge_subsets) <= set(spec.subsets)

    assert specs["voicebench"].judge_model == "gpt-4o-mini"
    assert specs["voicebench-ja"].judge_model == "gpt-4o-2024-08-06"
    assert specs["kvoicebench"].judge_model == "gpt-5.4"
    assert specs["kmmau"].judge_model == "gpt-5.4"


def test_spoken_question_benchmarks_do_not_leak_text() -> None:
    specs = load_specs(Path(__file__).parents[1] / "benchmarks.yaml")
    assert specs["voicebench"].question_mode == "audio_only"
    assert specs["kvoicebench"].question_mode == "audio_only"
    assert specs["voicebench-ja"].question_mode == "audio_only"


def test_default_runtime_covers_worst_case_long_multi_audio() -> None:
    server, _, client = _runtime(Path(__file__).parents[1] / "configs" / "default.yaml")
    assert server.max_model_len == 32_768
    assert server.audio_limit == 3
    assert client["max_model_len"] == 32_768
    assert client["max_audio_seconds"] == 720.0
