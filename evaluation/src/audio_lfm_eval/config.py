from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

QuestionMode = Literal["audio_only", "text_question"]
OfficialFormat = Literal[
    "voicebench_jsonl",
    "mmsu_jsonl",
    "mmau_json",
    "mmau_pro_parquet",
    "mmar_jsonl",
    "keval_jsonl",
    "flexeval_jsonl",
]


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    dataset_id: str
    revision: str
    subsets: tuple[str, ...]
    split: str
    question_mode: QuestionMode
    official_format: OfficialFormat
    audio_patterns: tuple[str, ...]
    scorer_repo: str
    scorer_revision: str
    supports_multiple_audio: bool = False
    judge_provider: str | None = None
    judge_model: str | None = None
    judge_subsets: tuple[str, ...] = ()


def _require_sha(value: object, *, field: str) -> str:
    rendered = str(value)
    if len(rendered) != 40 or any(char not in "0123456789abcdef" for char in rendered):
        raise ValueError(f"{field} must be an immutable 40-character SHA")
    return rendered


def load_specs(path: str | Path) -> dict[str, BenchmarkSpec]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("benchmarks"), dict):
        raise ValueError("Benchmark manifest must contain a benchmarks mapping")
    specs: dict[str, BenchmarkSpec] = {}
    for name, value in raw["benchmarks"].items():
        if not isinstance(value, dict):
            raise ValueError(f"Benchmark {name!r} must be a mapping")
        specs[str(name)] = BenchmarkSpec(
            name=str(name),
            dataset_id=str(value["dataset_id"]),
            revision=_require_sha(value["revision"], field=f"{name}.revision"),
            subsets=tuple(str(item) for item in value.get("subsets", ["default"])),
            split=str(value["split"]),
            question_mode=cast(QuestionMode, value["question_mode"]),
            official_format=cast(OfficialFormat, value["official_format"]),
            audio_patterns=tuple(str(item) for item in value["audio_patterns"]),
            scorer_repo=str(value["scorer_repo"]),
            scorer_revision=_require_sha(
                value["scorer_revision"], field=f"{name}.scorer_revision"
            ),
            supports_multiple_audio=bool(value.get("supports_multiple_audio", False)),
            judge_provider=(
                str(value["judge_provider"])
                if value.get("judge_provider") is not None
                else None
            ),
            judge_model=(
                str(value["judge_model"])
                if value.get("judge_model") is not None
                else None
            ),
            judge_subsets=tuple(str(item) for item in value.get("judge_subsets", [])),
        )
        spec = specs[str(name)]
        if bool(spec.judge_provider) != bool(spec.judge_model):
            raise ValueError(
                f"{name}.judge_provider and {name}.judge_model must be set together"
            )
        unknown_judge_subsets = set(spec.judge_subsets) - set(spec.subsets)
        if unknown_judge_subsets:
            raise ValueError(
                f"{name}.judge_subsets contains unknown subsets: "
                f"{sorted(unknown_judge_subsets)}"
            )
    return specs


def manifest_path() -> Path:
    return Path(__file__).resolve().parents[2] / "benchmarks.yaml"


def serializable_spec(spec: BenchmarkSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "dataset_id": spec.dataset_id,
        "revision": spec.revision,
        "subsets": list(spec.subsets),
        "split": spec.split,
        "question_mode": spec.question_mode,
        "official_format": spec.official_format,
        "audio_patterns": list(spec.audio_patterns),
        "scorer_repo": spec.scorer_repo,
        "scorer_revision": spec.scorer_revision,
        "supports_multiple_audio": spec.supports_multiple_audio,
        "judge_provider": spec.judge_provider,
        "judge_model": spec.judge_model,
        "judge_subsets": list(spec.judge_subsets),
    }
