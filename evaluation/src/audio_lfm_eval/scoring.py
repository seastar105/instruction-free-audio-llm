from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from audio_lfm_eval.config import BenchmarkSpec


def _predictions(
    path: str | Path,
    *,
    dedupe_by_id: bool = False,
) -> list[dict[str, Any]]:
    records_by_key: dict[tuple[str, object], dict[str, Any]] = {}
    ordered_keys: list[tuple[str, object]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if "error" not in record:
                    if not dedupe_by_id and record.get("row_index") is not None:
                        key = ("row_index", record["row_index"])
                    else:
                        key = ("id", record.get("id"))
                    if key not in records_by_key:
                        ordered_keys.append(key)
                    records_by_key[key] = record
    return [records_by_key[key] for key in ordered_keys]


def _merged(record: dict[str, Any], output_key: str) -> dict[str, Any]:
    source = dict(record["source_record"])
    source[output_key] = record["prediction"]
    return source


def _mmau_record(record: dict[str, Any]) -> dict[str, Any]:
    source = _merged(record, "model_output")
    attributes = source.pop("other_attributes", {})
    if isinstance(attributes, str):
        attributes = json.loads(attributes)
    if isinstance(attributes, dict):
        source.update(attributes)
    return source


def _flexeval_record(subset: str, record: dict[str, Any]) -> dict[str, Any]:
    source = record["source_record"]
    result: dict[str, Any] = {"lm_output": record["prediction"]}
    if subset in {"elyza", "spoken-elyza"}:
        result["references"] = [source["reference"]]
        result["task_inputs"] = {
            "messages": [{"role": "user", "content": source["text"]}],
            "eval_aspect": source["eval_aspect"],
        }
    elif subset == "m-ifeval":
        constraints = source["constraints"]
        result["references"] = []
        result["task_inputs"] = {
            "constraints": (
                json.loads(constraints) if isinstance(constraints, str) else constraints
            )
        }
    elif subset == "jamc-qa":
        result["references"] = [source["answer_choice"]]
        result["task_inputs"] = {"category": source["category"]}
    else:
        raise ValueError(f"Unsupported VoiceBench-JA subset {subset!r}")
    return result


def _kvoicebench_ifeval_metadata(data_root: Path) -> list[dict[str, Any]]:
    source_dir = data_root / "voicebench" / "ifeval"
    shards = sorted(source_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(
            "KVoiceBench IFEval scoring needs the original VoiceBench IFEval "
            f"metadata under {source_dir}"
        )
    table = pq.read_table(
        shards,
        columns=["key", "instruction_id_list", "kwargs"],
    )
    return cast(list[dict[str, Any]], table.to_pylist())


def _enrich_kvoicebench_ifeval_ground_truth(
    ground_truth: dict[str, Any],
    sample_id: str,
    source_metadata: list[dict[str, Any]],
) -> None:
    try:
        source_index = int(sample_id.rsplit("-", 1)[1])
        source = source_metadata[source_index]
    except (IndexError, ValueError) as error:
        raise ValueError(
            f"Cannot map KVoiceBench IFEval sample {sample_id!r} to its "
            "original VoiceBench row"
        ) from error
    ground_truth["instruction_id_list"] = source["instruction_id_list"]
    ground_truth["kwargs"] = source["kwargs"]
    ground_truth["original_voicebench_key"] = source["key"]


def materialize_official_input(
    *,
    spec: BenchmarkSpec,
    subset: str,
    predictions_path: str | Path,
    output_dir: str | Path,
    data_root: str | Path | None = None,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = _predictions(
        predictions_path,
        dedupe_by_id=spec.name == "voicebench",
    )
    if spec.official_format == "mmau_json":
        path = output / "official-input.json"
        payload = [_mmau_record(record) for record in records]
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path
    if spec.official_format == "mmau_pro_parquet":
        path = output / "official-input.parquet"
        payload = [_merged(record, "model_output") for record in records]
        pq.write_table(pa.Table.from_pylist(payload), path)
        return path
    path = output / "official-input.jsonl"
    if spec.official_format in {"voicebench_jsonl", "mmsu_jsonl"}:
        payload = [_merged(record, "response") for record in records]
    elif spec.official_format == "mmar_jsonl":
        payload = [_merged(record, "answer_prediction") for record in records]
    elif spec.official_format == "keval_jsonl":
        payload = [
            {"id": record["id"], "prediction": record["prediction"]}
            for record in records
        ]
        # KEval otherwise reloads the dataset's moving default branch. Preserve
        # the exact pinned rows used for generation and score against them via
        # ``--gt-jsonl`` so generation and scoring cannot drift apart.
        ground_truth_path = output / "ground-truth.jsonl"
        kvoice_ifeval_metadata = (
            _kvoicebench_ifeval_metadata(Path(data_root))
            if spec.name == "kvoicebench"
            and subset == "ifeval-test"
            and data_root is not None
            else None
        )
        with ground_truth_path.open("w", encoding="utf-8") as handle:
            for record in records:
                ground_truth = dict(record["source_record"])
                ground_truth["id"] = record["id"]
                if kvoice_ifeval_metadata is not None:
                    _enrich_kvoicebench_ifeval_ground_truth(
                        ground_truth,
                        str(record["id"]),
                        kvoice_ifeval_metadata,
                    )
                handle.write(json.dumps(ground_truth, ensure_ascii=False) + "\n")
    elif spec.official_format == "flexeval_jsonl":
        payload = [_flexeval_record(subset, record) for record in records]
    else:
        raise ValueError(f"Unsupported official format {spec.official_format!r}")
    with path.open("w", encoding="utf-8") as handle:
        for record in payload:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def voicebench_evaluator(subset: str) -> str:
    if subset in {"alpacaeval", "alpacaeval_full", "commoneval", "wildvoice"}:
        return "open"
    if subset == "sd-qa":
        return "qa"
    if subset in {"mmsu", "openbookqa"}:
        return "mcq"
    if subset == "bbh":
        return "bbh"
    if subset == "ifeval":
        return "ifeval"
    if subset == "advbench":
        return "harm"
    raise ValueError(f"No VoiceBench evaluator for {subset!r}")


def scorer_plan(
    *,
    spec: BenchmarkSpec,
    subset: str,
    scorer_root: str | Path,
    input_path: str | Path,
    data_root: str | Path,
) -> list[list[str]]:
    root = Path(scorer_root).resolve() / spec.name
    source = str(Path(input_path).resolve())
    if spec.name == "voicebench":
        evaluator = voicebench_evaluator(subset)
        local_source = Path(input_path).name
        if evaluator in {"open", "qa"}:
            if evaluator == "qa":
                safe_qa_command = [
                    "python",
                    "-m",
                    "audio_lfm_eval.voicebench_sdqa",
                    "--input",
                    f"result-{local_source}",
                    "--output",
                    "scores.json",
                ]
            else:
                safe_qa_command = [
                    "python",
                    str(root / "evaluate.py"),
                    "--src_file",
                    f"result-{local_source}",
                    "--evaluator",
                    evaluator,
                ]
            return [
                [
                    "python",
                    "-m",
                    "audio_lfm_eval.voicebench_api_judge",
                    "--src_file",
                    local_source,
                ],
                safe_qa_command,
            ]
        return [
            [
                "python",
                str(root / "evaluate.py"),
                "--src_file",
                source,
                "--evaluator",
                evaluator,
            ]
        ]
    if spec.name == "mmsu":
        return [["python", str(root / "mmsu_evaluation.py"), source]]
    if spec.name == "mmau":
        return [["python", str(root / "evaluation.py"), "--input", source]]
    if spec.name == "mmar":
        return [["python", str(root / "code/evaluation.py"), "--input", source]]
    if spec.name == "mmau-pro":
        judge_output = str(
            Path(input_path).with_name("open-judge-vllm.jsonl").resolve()
        )
        return [
            [
                sys.executable,
                "-m",
                "audio_lfm_eval.mmau_pro_vllm_judge",
                source,
                "--output",
                judge_output,
            ],
            [
                "python",
                str(root / "evaluate_mmau_pro_comprehensive.py"),
                source,
                "--model_output_column",
                "model_output",
                "--open_judge_results",
                judge_output,
                "--skip_closed",
            ],
        ]
    if spec.name in {"kvoicebench", "kmmau"}:
        benchmark = spec.name
        command = [
            "keval",
            "evaluate",
            "--benchmark",
            benchmark,
            "--subset",
            subset,
            "--predictions",
            source,
            "--gt-jsonl",
            str(Path(input_path).with_name("ground-truth.jsonl").resolve()),
            "--output",
            str(Path(input_path).with_name("scores.json").resolve()),
        ]
        if subset in spec.judge_subsets:
            if spec.judge_provider is None or spec.judge_model is None:
                raise ValueError(
                    f"No pinned judge configuration for {benchmark}:{subset}"
                )
            command.extend(
                [
                    "--judge-provider",
                    spec.judge_provider,
                    "--judge-model",
                    spec.judge_model,
                ]
            )
        return [command]
    if spec.name == "voicebench-ja":
        if subset == "m-ifeval":
            metrics_path = (
                Path(__file__).resolve().parents[2]
                / "configs/voicebench-ja-m-ifeval-flexeval-metrics.jsonnet"
            )
        else:
            metrics_path = (
                Path(data_root) / spec.name / f"{subset}-flexeval-metrics.jsonnet"
            )
        metrics = str(metrics_path.resolve())
        return [
            [
                "flexeval_file",
                "--eval_file",
                source,
                "--metrics",
                metrics,
                "--save_dir",
                str(Path(input_path).with_name("flexeval-scores").resolve()),
                "--force",
                "true",
            ]
        ]
    raise ValueError(f"No scorer plan for {spec.name!r} in {root}")
