from __future__ import annotations

import subprocess
import tarfile
import zipfile
from pathlib import Path

from audio_lfm_eval.config import BenchmarkSpec

_SCORER_PATCHES = {
    "kmmau": "raon-eval.patch",
    "kvoicebench": "raon-eval.patch",
    "mmau-pro": "mmau-pro.patch",
    "mmsu": "mmsu.patch",
    "voicebench": "voicebench.patch",
    "voicebench-ja": "voicebench-ja.patch",
}


def _apply_scorer_patch(
    destination: Path,
    benchmark: str,
    *,
    git_executable: str = "git",
    patch_root: Path | None = None,
) -> None:
    patch_name = _SCORER_PATCHES.get(benchmark)
    if patch_name is None:
        return
    patch_dir = patch_root or Path(__file__).with_name("scorer_patches")
    patch = patch_dir / patch_name
    if not patch.is_file():
        raise FileNotFoundError(f"Missing scorer patch: {patch}")
    check = subprocess.run(
        [
            git_executable,
            "-C",
            str(destination),
            "apply",
            "--ignore-space-change",
            "--check",
            str(patch),
        ],
        capture_output=True,
        text=True,
    )
    if check.returncode == 0:
        subprocess.run(
            [
                git_executable,
                "-C",
                str(destination),
                "apply",
                "--ignore-space-change",
                str(patch),
            ],
            check=True,
        )
        return
    reverse = subprocess.run(
        [
            git_executable,
            "-C",
            str(destination),
            "apply",
            "--ignore-space-change",
            "--reverse",
            "--check",
            str(patch),
        ],
        capture_output=True,
        text=True,
    )
    if reverse.returncode != 0:
        raise RuntimeError(
            f"Cannot apply scorer patch {patch_name} to {destination}: "
            f"{check.stderr.strip()}"
        )


def download_snapshot(
    *,
    spec: BenchmarkSpec,
    subset: str,
    output_root: str | Path,
    hf_executable: str = "hf",
) -> Path:
    if subset not in spec.subsets:
        raise ValueError(f"Unknown {spec.name} subset {subset!r}")
    destination = Path(output_root) / spec.name
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        hf_executable,
        "download",
        spec.dataset_id,
        "--repo-type",
        "dataset",
        "--revision",
        spec.revision,
        "--local-dir",
        str(destination),
        "--max-workers",
        "2",
    ]
    for pattern in spec.audio_patterns:
        command.extend(["--include", pattern.format(subset=subset)])
    subprocess.run(command, check=True)
    return destination


def sync_scorer_source(
    *, spec: BenchmarkSpec, output_root: str | Path, git_executable: str = "git"
) -> Path:
    destination = Path(output_root) / spec.name
    if not destination.exists():
        subprocess.run(
            [
                git_executable,
                "clone",
                "--filter=blob:none",
                spec.scorer_repo,
                str(destination),
            ],
            check=True,
        )
    subprocess.run(
        [
            git_executable,
            "-C",
            str(destination),
            "fetch",
            "--depth",
            "1",
            "origin",
            spec.scorer_revision,
        ],
        check=True,
    )
    subprocess.run(
        [
            git_executable,
            "-C",
            str(destination),
            "checkout",
            "--detach",
            spec.scorer_revision,
        ],
        check=True,
    )
    _apply_scorer_patch(destination, spec.name, git_executable=git_executable)
    return destination


def _validated_target(root: Path, member: str) -> Path:
    target = (root / member).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Archive member escapes destination: {member!r}")
    return target


def unpack_benchmark(spec: BenchmarkSpec, data_root: str | Path) -> list[Path]:
    root = Path(data_root) / spec.name
    extracted: list[Path] = []
    for archive in sorted(root.glob("*.zip")):
        with zipfile.ZipFile(archive) as handle:
            for zip_member in handle.infolist():
                _validated_target(root, zip_member.filename)
            handle.extractall(root)
        extracted.append(archive)
    for archive in sorted(root.glob("*.tar.gz")):
        with tarfile.open(archive) as handle:
            for tar_member in handle.getmembers():
                _validated_target(root, tar_member.name)
                if tar_member.issym() or tar_member.islnk():
                    raise ValueError(
                        f"Archive links are not allowed: {tar_member.name!r}"
                    )
            handle.extractall(root)
        extracted.append(archive)
    return extracted
