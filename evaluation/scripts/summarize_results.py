#!/usr/bin/env python3
"""Build compact, publication-safe summaries of completed evaluation runs."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

VOICEBENCH_SUBSETS = (
    "alpacaeval_full",
    "commoneval",
    "wildvoice",
    "sd-qa",
    "mmsu",
    "openbookqa",
    "bbh",
    "ifeval",
    "advbench",
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _last_info_dict(path: Path) -> dict[str, Any]:
    values: list[dict[str, Any]] = []
    for line in path.read_text(errors="replace").splitlines():
        if " | INFO " not in line or " - {" not in line:
            continue
        value = ast.literal_eval(line.split(" - ", 1)[1])
        if isinstance(value, dict):
            values.append(value)
    if not values:
        raise ValueError(f"No scorer result dictionary in {path}")
    return values[-1]


def _last_match(path: Path, pattern: str) -> re.Match[str]:
    matches = list(re.finditer(pattern, path.read_text(errors="replace")))
    if not matches:
        raise ValueError(f"Pattern {pattern!r} not found in {path}")
    return matches[-1]


def _aggregate_scores(run: Path, benchmark: str) -> dict[str, Any]:
    subsets: dict[str, Any] = {}
    weighted_sum = 0.0
    weighted_count = 0
    for path in sorted((run / benchmark).glob("*/scoring/scores.json")):
        payload = _json(path)
        metrics = payload["aggregate_metrics"]
        subsets[path.parents[1].name] = metrics
        if benchmark == "kmmau":
            accuracy = metrics["accuracy"]
            weighted_sum += float(accuracy["value"]) * int(accuracy["count"])
            weighted_count += int(accuracy["count"])
    result: dict[str, Any] = {"subsets": subsets}
    if benchmark == "kmmau":
        result["weighted_accuracy"] = weighted_sum / weighted_count
        result["num_samples"] = weighted_count
    return result


def _voicebench(run: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for subset in VOICEBENCH_SUBSETS:
        scoring = run / "voicebench" / subset / "scoring"
        if subset == "sd-qa":
            payload = _json(scoring / "scores.json")
            result[subset] = payload.get("aggregate_metrics", payload)
        else:
            result[subset] = _last_info_dict(scoring / "scorer-output.log")
    return result


def _voicebench_ja(run: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    root = run / "voicebench-ja"
    for path in sorted(root.glob("*/scoring/flexeval-scores/metrics.json")):
        metrics = _json(path)
        metrics.pop("elapsed_time", None)
        result[path.parents[2].name] = metrics
    return result


def _classic_score(run: Path, benchmark: str) -> dict[str, Any]:
    log = run / benchmark / "default" / "scoring" / "scorer-output.log"
    if benchmark == "mmsu":
        accuracy = float(_last_match(log, r"Overall Accuracy: ([0-9.]+)").group(1))
        count = int(_last_match(log, r"Total count: (\d+)").group(1))
        malformed = int(
            _last_match(
                log, r"Unparseable responses counted as incorrect: (\d+)"
            ).group(1)
        )
        return {
            "accuracy": accuracy,
            "num_samples": count,
            "unparseable_counted_as_incorrect": malformed,
        }
    match = _last_match(log, r"Total Accuracy: ([0-9.]+)% over (\d+) samples")
    return {
        "accuracy": float(match.group(1)) / 100.0,
        "num_samples": int(match.group(2)),
    }


def _mmau_pro(run: Path) -> dict[str, Any]:
    payload = _json(
        run
        / "mmau-pro"
        / "default"
        / "scoring"
        / "official-input_comprehensive_results.json"
    )
    summary = payload["evaluation_summary"]
    return {
        "weighted_score": summary["overall_weighted_performance"],
        "evaluation_scope": summary["evaluation_scope"],
        "evaluated_samples": summary["evaluated_samples"],
        "total_samples": summary["total_samples"],
        "omitted_closed_samples": summary["omitted_closed_samples"],
        "categories": payload["category_results"],
        "omitted_official_metrics": ["closed_ended_nvembed"],
    }


def summarize(root: Path, tag: str) -> dict[str, Any]:
    run = root / "evaluation-runs" / tag
    generation = _json(run / "mmau" / "default" / "generation_manifest.json")
    return {
        "format_version": 1,
        "checkpoint": tag,
        "model": generation["model"],
        "generation": generation["generation"],
        "transport": generation["transport"],
        "scoring_policy": {
            "unparseable_model_responses": "count_as_zero_in_full_denominator",
            "unparseable_judge_responses": "count_as_zero_in_full_denominator",
            "voicebench_sdqa_panda": "omitted_due_to_mutable_pickle_deserialization",
            "mmau_pro_closed_ended_nvembed": "omitted_by_request",
        },
        "judges": {
            "mmau_pro_open": {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "backend": "vllm",
            },
            "voicebench_open_and_sdqa": {
                "model": "gpt-4o-mini",
                "backend": "openai_api",
            },
            "kvoicebench_and_kmmau": {
                "model": "gpt-5.4",
                "backend": "openai_api",
            },
        },
        "benchmarks": {
            "mmau": _classic_score(run, "mmau"),
            "mmsu": _classic_score(run, "mmsu"),
            "mmau_pro": _mmau_pro(run),
            "mmar": _classic_score(run, "mmar"),
            "voicebench": _voicebench(run),
            "kvoicebench": _aggregate_scores(run, "kvoicebench"),
            "kmmau": _aggregate_scores(run, "kmmau"),
            "voicebench_ja": _voicebench_ja(run),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--tags", nargs="+", default=("6k", "20k"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for tag in args.tags:
        output = output_dir / f"evaluation-{tag}.json"
        output.write_text(json.dumps(summarize(root, tag), indent=2) + "\n")
        print(output.relative_to(root))


if __name__ == "__main__":
    main()
