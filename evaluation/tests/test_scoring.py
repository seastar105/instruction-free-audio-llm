import json
import sys
from pathlib import Path

from audio_lfm_eval.config import load_specs
from audio_lfm_eval.scoring import _predictions, materialize_official_input, scorer_plan
from audio_lfm_eval.voicebench_sdqa import score_gpt_judgments


def _prediction(path: Path, source: dict[str, object]) -> None:
    path.write_text(
        json.dumps({"id": "id-1", "prediction": "A. Rain", "source_record": source})
        + "\n",
        encoding="utf-8",
    )


def test_scoring_deduplicates_successful_resume_rows(tmp_path: Path) -> None:
    source = tmp_path / "predictions.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "row_index": 0, "prediction": "old"}),
                json.dumps({"id": "a", "row_index": 0, "error": "retry"}),
                json.dumps({"id": "a", "row_index": 0, "prediction": "new"}),
                json.dumps({"id": "a", "row_index": 1, "prediction": "other"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert [record["prediction"] for record in _predictions(source)] == [
        "new",
        "other",
    ]
    assert [
        record["prediction"] for record in _predictions(source, dedupe_by_id=True)
    ] == ["other"]


def test_mmar_official_field(tmp_path) -> None:
    specs = load_specs(Path(__file__).parents[1] / "benchmarks.yaml")
    predictions = tmp_path / "predictions.jsonl"
    _prediction(predictions, {"id": "id-1", "answer": "Rain"})
    result = materialize_official_input(
        spec=specs["mmar"],
        subset="default",
        predictions_path=predictions,
        output_dir=tmp_path / "scores",
    )
    record = json.loads(result.read_text(encoding="utf-8"))
    assert record["answer_prediction"] == "A. Rain"


def test_keval_format_contains_only_public_contract(tmp_path) -> None:
    specs = load_specs(Path(__file__).parents[1] / "benchmarks.yaml")
    predictions = tmp_path / "predictions.jsonl"
    _prediction(predictions, {"id": "id-1", "answer": "Rain"})
    result = materialize_official_input(
        spec=specs["kmmau"],
        subset="age",
        predictions_path=predictions,
        output_dir=tmp_path / "scores",
    )
    assert json.loads(result.read_text(encoding="utf-8")) == {
        "id": "id-1",
        "prediction": "A. Rain",
    }
    assert json.loads(result.with_name("ground-truth.jsonl").read_text()) == {
        "id": "id-1",
        "answer": "Rain",
    }


def test_kvoicebench_ifeval_restores_original_constraint_metadata(
    tmp_path: Path,
) -> None:
    specs = load_specs(Path(__file__).parents[1] / "benchmarks.yaml")
    source_dir = tmp_path / "data" / "voicebench" / "ifeval"
    source_dir.mkdir(parents=True)
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "key": 1001,
                    "instruction_id_list": ["punctuation:no_comma"],
                    "kwargs": [{"unused": None}],
                }
            ]
        ),
        source_dir / "test.parquet",
    )
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "id": "ifeval-test-00000",
                "prediction": "response without comma",
                "source_record": {
                    "id": "ifeval-test-00000",
                    "answer": "",
                    "transcription": "translated prompt",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = materialize_official_input(
        spec=specs["kvoicebench"],
        subset="ifeval-test",
        predictions_path=predictions,
        output_dir=tmp_path / "scores",
        data_root=tmp_path / "data",
    )
    ground_truth = json.loads(
        result.with_name("ground-truth.jsonl").read_text(encoding="utf-8")
    )
    assert ground_truth["instruction_id_list"] == ["punctuation:no_comma"]
    assert ground_truth["kwargs"] == [{"unused": None}]
    assert ground_truth["original_voicebench_key"] == 1001


def test_mmau_flattens_official_attributes(tmp_path) -> None:
    specs = load_specs(Path(__file__).parents[1] / "benchmarks.yaml")
    predictions = tmp_path / "predictions.jsonl"
    _prediction(
        predictions,
        {
            "answer": "(A) Rain",
            "choices": ["(A) Rain", "(B) Wind"],
            "other_attributes": json.dumps(
                {"id": "official-id", "task": "sound", "difficulty": "easy"}
            ),
        },
    )
    result = materialize_official_input(
        spec=specs["mmau"],
        subset="default",
        predictions_path=predictions,
        output_dir=tmp_path / "scores",
    )
    record = json.loads(result.read_text(encoding="utf-8"))[0]
    assert record["id"] == "official-id"
    assert record["task"] == "sound"
    assert "other_attributes" not in record


def test_kvoicebench_pins_paper_comparable_judge_model(tmp_path) -> None:
    specs = load_specs(Path(__file__).parents[1] / "benchmarks.yaml")
    plan = scorer_plan(
        spec=specs["kvoicebench"],
        subset="alpacaeval_full-test",
        scorer_root=tmp_path,
        input_path=tmp_path / "official-input.jsonl",
        data_root=tmp_path,
    )
    assert plan[0][-4:] == [
        "--judge-provider",
        "openai",
        "--judge-model",
        "gpt-5.4",
    ]


def test_kvoicebench_programmatic_subset_has_no_judge_flags(tmp_path) -> None:
    specs = load_specs(Path(__file__).parents[1] / "benchmarks.yaml")
    plan = scorer_plan(
        spec=specs["kvoicebench"],
        subset="ifeval-test",
        scorer_root=tmp_path,
        input_path=tmp_path / "official-input.jsonl",
        data_root=tmp_path,
    )
    assert "--judge-model" not in plan[0]
    assert plan[0][plan[0].index("--gt-jsonl") + 1] == str(
        (tmp_path / "ground-truth.jsonl").resolve()
    )
    assert plan[0][plan[0].index("--output") + 1] == str(
        (tmp_path / "scores.json").resolve()
    )


def test_voicebench_ja_plan_uses_absolute_runtime_paths(tmp_path) -> None:
    specs = load_specs(Path(__file__).parents[1] / "benchmarks.yaml")
    plan = scorer_plan(
        spec=specs["voicebench-ja"],
        subset="m-ifeval",
        scorer_root=tmp_path,
        input_path=tmp_path / "scores" / "official-input.jsonl",
        data_root=tmp_path / "data",
    )[0]
    metrics = Path(plan[plan.index("--metrics") + 1])
    assert metrics.name == "voicebench-ja-m-ifeval-flexeval-metrics.jsonnet"
    assert metrics.is_absolute()
    assert plan[plan.index("--save_dir") + 1] == str(
        (tmp_path / "scores/flexeval-scores").resolve()
    )
    assert plan[plan.index("--force") + 1] == "true"


def test_voicebench_sdqa_uses_safe_gpt_only_scorer(tmp_path) -> None:
    specs = load_specs(Path(__file__).parents[1] / "benchmarks.yaml")
    plan = scorer_plan(
        spec=specs["voicebench"],
        subset="sd-qa",
        scorer_root=tmp_path,
        input_path=tmp_path / "official-input.jsonl",
        data_root=tmp_path,
    )
    assert plan[0][1:3] == ["-m", "audio_lfm_eval.voicebench_api_judge"]
    assert plan[1][1:3] == ["-m", "audio_lfm_eval.voicebench_sdqa"]
    assert "evaluate.py" not in " ".join(plan[1])


def test_mmau_pro_uses_batched_vllm_judge_before_native_embedding(tmp_path) -> None:
    specs = load_specs(Path(__file__).parents[1] / "benchmarks.yaml")
    input_path = tmp_path / "scores" / "official-input.parquet"
    plan = scorer_plan(
        spec=specs["mmau-pro"],
        subset="default",
        scorer_root=tmp_path,
        input_path=input_path,
        data_root=tmp_path,
    )
    judge_output = str(input_path.with_name("open-judge-vllm.jsonl").resolve())
    assert plan[0] == [
        sys.executable,
        "-m",
        "audio_lfm_eval.mmau_pro_vllm_judge",
        str(input_path.resolve()),
        "--output",
        judge_output,
    ]
    assert plan[1][plan[1].index("--open_judge_results") + 1] == judge_output
    assert "--skip_closed" in plan[1]


def test_safe_sdqa_scorer_marks_panda_omitted(tmp_path) -> None:
    source = tmp_path / "judgments.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "score": ["Yes", "yes", "No"]}),
                json.dumps({"id": "b", "score": ["No", "No", "Yes"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result = score_gpt_judgments(source)
    assert result["gpt"] == 50.0
    assert result["panda"] is None
    assert result["complete_official_sdqa"] is False
