from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_SHA = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class ReferenceRange:
    expected: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model_id: str
    revision: str
    architecture: str
    language: str
    vllm_supported: bool
    unsupported_reason: str
    references: dict[str, ReferenceRange]

    def require_vllm_support(self) -> None:
        if not self.vllm_supported:
            raise RuntimeError(
                f"{self.name} cannot be launched by the pinned vLLM runtime: "
                f"{self.unsupported_reason}"
            )


def profile_manifest_path() -> Path:
    return Path(__file__).resolve().parents[2] / "model_profiles.yaml"


def load_model_profiles(path: str | Path | None = None) -> dict[str, ModelProfile]:
    source = Path(path) if path is not None else profile_manifest_path()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    profiles: dict[str, ModelProfile] = {}
    for name, item in raw["profiles"].items():
        revision = str(item["revision"])
        if not _SHA.fullmatch(revision):
            raise ValueError(f"{name} revision must be an immutable 40-character SHA")
        serving = item["serving"]
        references = {
            metric: ReferenceRange(
                expected=float(values["expected"]),
                minimum=float(values["minimum"]),
                maximum=float(values["maximum"]),
            )
            for metric, values in item.get("references", {}).items()
        }
        profiles[name] = ModelProfile(
            name=name,
            model_id=str(item["model_id"]),
            revision=revision,
            architecture=str(item["architecture"]),
            language=str(item["language"]),
            vllm_supported=bool(serving["supported"]),
            unsupported_reason=str(serving["reason"]),
            references=references,
        )
    return profiles


def validate_reference_scores(
    profile: ModelProfile, scores: dict[str, float]
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for metric, reference in profile.references.items():
        if metric not in scores:
            results[metric] = {"status": "missing", "expected": reference.expected}
            continue
        value = float(scores[metric])
        results[metric] = {
            "status": "pass"
            if reference.minimum <= value <= reference.maximum
            else "fail",
            "value": value,
            "expected": reference.expected,
            "minimum": reference.minimum,
            "maximum": reference.maximum,
        }
    return results


def load_score_summary(path: str | Path) -> dict[str, float]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("score summary must be a JSON object")
    return {str(key): float(value) for key, value in raw.items()}
