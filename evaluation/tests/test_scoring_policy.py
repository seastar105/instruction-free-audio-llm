from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from audio_lfm_eval.mmau_pro_vllm_judge import (
    build_conversations,
    completed_result_count,
)


def test_mmau_pro_vllm_judge_keeps_every_open_response() -> None:
    records = [
        {
            "question": "Question one?",
            "answer": "Reference one",
            "model_output": "Response one",
        },
        {
            "question": "Question two?",
            "answer": "Reference two",
            "model_output": "",
        },
    ]
    conversations = build_conversations(records)
    assert len(conversations) == 2
    assert "Model Response: Response one" in conversations[0][1]["content"]
    assert "Model Response: \n\n" in conversations[1][1]["content"]


def test_mmau_pro_vllm_judge_reuses_only_complete_results(tmp_path: Path) -> None:
    output = tmp_path / "judge.jsonl"
    output.write_text(
        "\n".join(
            [
                json.dumps({"open_index": 0, "evaluation": "valid"}),
                json.dumps({"open_index": 1, "evaluation": ""}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert completed_result_count(output, 2) == 2
    assert completed_result_count(output, 3) is None

    output.write_text(
        json.dumps({"open_index": 1, "evaluation": "out of order"}) + "\n",
        encoding="utf-8",
    )
    assert completed_result_count(output, 1) is None


def test_standalone_mmsu_counts_unparseable_response_as_incorrect(
    tmp_path: Path,
) -> None:
    source = tmp_path / "mmsu.jsonl"
    common = {
        "category": "test",
        "sub-category": "test",
        "answer_gt": "rain",
        "choice_a": "rain",
        "choice_b": "snow",
        "choice_c": "wind",
        "choice_d": "sun",
    }
    source.write_text(
        "\n".join(
            [
                json.dumps({**common, "response": "A"}),
                json.dumps({**common, "response": "not a choice"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    scorer = (
        Path(__file__).parents[2] / "evaluation-scorers" / "mmsu" / "mmsu_evaluation.py"
    )
    completed = subprocess.run(
        [sys.executable, str(scorer), str(source)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Overall Accuracy: 0.5000" in completed.stdout
    assert "Total count: 2" in completed.stdout
    assert "Unparseable responses counted as incorrect: 1" in completed.stdout


def test_voicebench_never_randomly_guesses_unparseable_responses() -> None:
    repository_root = Path(__file__).parents[2]
    scorer_root = repository_root / "evaluation-scorers" / "voicebench"
    scorer_python = repository_root / ".venv-evaluation-scorers" / "bin" / "python"
    code = """
from src.evaluator.bbh import BBHEvaluator
from src.evaluator.mcq import MCQEvaluator
from src.evaluator.open import OpenEvaluator

mcq = MCQEvaluator().evaluate([{"reference": "A", "response": "unparseable"}])
assert mcq == {"acc": 0.0, "fail": 100.0}
bbh = BBHEvaluator().evaluate([
    {
        "id": "sports_understanding-example",
        "reference": "yes",
        "response": "unparseable",
    }
])
assert bbh == {"acc": 0.0}
assert OpenEvaluator().evaluate([{"score": ["unparseable"]}]) == {"gpt": 0.0}
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(scorer_root)
    subprocess.run(
        [str(scorer_python), "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_voicebench_sdqa_deduplicates_resumed_judge_rows(tmp_path: Path) -> None:
    from audio_lfm_eval.voicebench_sdqa import score_gpt_judgments

    source = tmp_path / "judge.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "a", "reference": "x", "score": ["No"]}),
                json.dumps({"prompt": "b", "reference": "y", "score": ["No"]}),
                json.dumps(
                    {
                        "prompt": "a",
                        "reference": "x",
                        "score": ["Yes", "Yes", "No"],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result = score_gpt_judgments(source)
    assert result["num_samples"] == 2
    assert result["gpt"] == 50.0
