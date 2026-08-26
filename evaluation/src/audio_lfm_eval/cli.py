from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

from audio_lfm_eval.config import load_specs, manifest_path, serializable_spec
from audio_lfm_eval.downloads import (
    download_snapshot,
    sync_scorer_source,
    unpack_benchmark,
)
from audio_lfm_eval.generation import generate_benchmark
from audio_lfm_eval.http_client import GenerationSettings, VllmHttpClient
from audio_lfm_eval.model_profiles import (
    load_model_profiles,
    load_score_summary,
    validate_reference_scores,
)
from audio_lfm_eval.scoring import materialize_official_input, scorer_plan
from audio_lfm_eval.server import (
    ServerProcess,
    ServerSettings,
    build_server_command,
)

app = typer.Typer(no_args_is_help=True)
DEFAULT_MANIFEST = manifest_path()
DEFAULT_CONFIG = DEFAULT_MANIFEST.parent / "configs/default.yaml"


@app.command("list-model-profiles")
def list_model_profiles() -> None:
    """List pinned smoke/reference models and native-vLLM capability."""
    profiles = load_model_profiles()
    payload = {
        name: {
            "model_id": profile.model_id,
            "revision": profile.revision,
            "architecture": profile.architecture,
            "vllm_supported": profile.vllm_supported,
            "reason": profile.unsupported_reason,
        }
        for name, profile in profiles.items()
    }
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("preflight-model-profile")
def preflight_model_profile(profile: Annotated[str, typer.Option("--profile")]) -> None:
    """Fail early when a reference checkpoint lacks the required vLLM adapter."""
    profiles = load_model_profiles()
    if profile not in profiles:
        raise typer.BadParameter(f"Unknown model profile {profile!r}")
    selected = profiles[profile]
    try:
        selected.require_vllm_support()
    except RuntimeError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"{selected.model_id}@{selected.revision} supports vLLM HTTP serving")


@app.command("validate-reference")
def validate_reference(
    profile: Annotated[str, typer.Option("--profile")],
    scores: Annotated[Path, typer.Option("--scores", exists=True)],
    require_complete: Annotated[bool, typer.Option("--require-complete")] = False,
) -> None:
    """Compare a canonical stage-2 score JSON with published numerical ranges."""
    selected = load_model_profiles()[profile]
    result = validate_reference_scores(selected, load_score_summary(scores))
    typer.echo(json.dumps(result, indent=2, sort_keys=True))
    bad = [
        metric
        for metric, item in result.items()
        if item["status"] == "fail"
        or (require_complete and item["status"] == "missing")
    ]
    if bad:
        raise typer.Exit(code=1)


def _runtime(path: Path) -> tuple[ServerSettings, GenerationSettings, dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    server = ServerSettings(**raw["server"])
    generation = GenerationSettings(**raw["generation"])
    client = dict(raw["client"])
    if int(client["max_model_len"]) != server.max_model_len:
        raise ValueError(
            "client.max_model_len must exactly match server.max_model_len so "
            "full-audio context preflight cannot diverge from vLLM"
        )
    if server.audio_limit < 3:
        raise ValueError(
            "server.audio_limit must be at least 3 for full MMAU-Pro coverage"
        )
    if float(client["max_audio_seconds"]) != server.max_audio_decode_duration_seconds:
        raise ValueError(
            "client.max_audio_seconds must exactly match "
            "server.max_audio_decode_duration_seconds"
        )
    return server, generation, client


def _vllm_executable(value: str) -> str:
    if value != "vllm" or shutil.which(value) is not None:
        return value
    sibling = Path(sys.executable).with_name("vllm")
    return str(sibling) if sibling.exists() else value


def _selected(value: str, specs: dict[str, Any]) -> list[tuple[str, str]]:
    name, separator, subset = value.partition(":")
    if name not in specs:
        raise typer.BadParameter(f"Unknown benchmark {name!r}")
    if separator:
        return [(name, subset)]
    return [(name, item) for item in specs[name].subsets]


@app.command("list")
def list_benchmarks(
    manifest: Annotated[Path, typer.Option("--manifest")] = DEFAULT_MANIFEST,
) -> None:
    """List pinned datasets, subsets, and scorer revisions."""
    payload = {
        name: serializable_spec(spec) for name, spec in load_specs(manifest).items()
    }
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


@app.command()
def download(
    benchmark: Annotated[str, typer.Option("--benchmark")],
    subset: Annotated[str, typer.Option("--subset")],
    output_root: Annotated[Path, typer.Option("--output-root")] = Path(
        "evaluation-data"
    ),
    hf_executable: Annotated[str, typer.Option("--hf-executable")] = "hf",
    manifest: Annotated[Path, typer.Option("--manifest")] = DEFAULT_MANIFEST,
) -> None:
    """Download one immutable dataset subset through the hf CLI."""
    spec = load_specs(manifest)[benchmark]
    destination = download_snapshot(
        spec=spec,
        subset=subset,
        output_root=output_root,
        hf_executable=hf_executable,
    )
    typer.echo(destination)


@app.command("sync-scorers")
def sync_scorers(
    benchmarks: Annotated[list[str] | None, typer.Option("--benchmark")] = None,
    output_root: Annotated[Path, typer.Option("--output-root")] = Path(
        "evaluation-scorers"
    ),
    manifest: Annotated[Path, typer.Option("--manifest")] = DEFAULT_MANIFEST,
) -> None:
    """Clone official scorer source at the manifest's immutable commits."""
    specs = load_specs(manifest)
    names = benchmarks or list(specs)
    for name in names:
        typer.echo(sync_scorer_source(spec=specs[name], output_root=output_root))


@app.command()
def unpack(
    benchmark: Annotated[str, typer.Option("--benchmark")],
    data_root: Annotated[Path, typer.Option("--data-root", exists=True)] = Path(
        "evaluation-data"
    ),
    manifest: Annotated[Path, typer.Option("--manifest")] = DEFAULT_MANIFEST,
) -> None:
    """Safely unpack benchmark-owned zip/tar archives in their snapshot directory."""
    spec = load_specs(manifest)[benchmark]
    for archive in unpack_benchmark(spec, data_root):
        typer.echo(f"unpacked {archive}")


def _generate_selected(
    *,
    client: VllmHttpClient,
    selections: list[str],
    specs: dict[str, Any],
    model_name: str,
    model_identity: dict[str, str],
    data_root: Path,
    output_root: Path,
    generation: GenerationSettings,
    client_config: dict[str, Any],
    limit: int | None,
) -> None:
    for selection in selections:
        for name, subset in _selected(selection, specs):
            progress = generate_benchmark(
                client=client,
                model=model_name,
                model_identity=model_identity,
                spec=specs[name],
                subset=subset,
                data_root=data_root / name,
                output_root=output_root,
                settings=generation,
                concurrency=int(client_config["request_concurrency"]),
                max_audio_seconds=float(client_config["max_audio_seconds"]),
                max_model_len=int(client_config["max_model_len"]),
                audio_chunk_seconds=float(client_config["audio_chunk_seconds"]),
                audio_stack_factor=int(client_config["audio_stack_factor"]),
                text_prompt_token_reserve=int(
                    client_config["text_prompt_token_reserve"]
                ),
                max_inflight_audio_seconds=float(
                    client_config["max_inflight_audio_seconds"]
                ),
                limit=limit,
            )
            typer.echo(f"{name}:{subset} {json.dumps(progress, sort_keys=True)}")


@app.command()
def generate(
    benchmarks: Annotated[list[str], typer.Option("--benchmark")],
    model_export: Annotated[Path, typer.Option("--model-export", exists=True)],
    data_root: Annotated[Path, typer.Option("--data-root", exists=True)],
    output_root: Annotated[Path, typer.Option("--output-root")] = Path(
        "evaluation-runs"
    ),
    model_name: Annotated[str, typer.Option("--model-name")] = "audio-lfm-eval",
    base_url: Annotated[str, typer.Option("--base-url")] = "http://127.0.0.1:8000",
    config: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
    manifest: Annotated[Path, typer.Option("--manifest")] = DEFAULT_MANIFEST,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
) -> None:
    """Run selected benchmarks against an already-running vLLM HTTP server."""
    _, generation_config, client_config = _runtime(config)
    specs = load_specs(manifest)
    client = VllmHttpClient(
        base_url,
        timeout_seconds=float(client_config["request_timeout_seconds"]),
        max_connections=int(client_config["http_max_connections"]),
    )
    try:
        if not client.health():
            raise typer.BadParameter(f"No healthy vLLM server at {base_url}")
        _generate_selected(
            client=client,
            selections=benchmarks,
            specs=specs,
            model_name=model_name,
            model_identity={
                "kind": "projector-export",
                "config_sha256": hashlib.sha256(
                    (model_export / "config.json").read_bytes()
                ).hexdigest(),
            },
            data_root=data_root,
            output_root=output_root,
            generation=generation_config,
            client_config=client_config,
            limit=limit,
        )
    finally:
        client.close()


@app.command("run-suite")
def run_suite(
    benchmarks: Annotated[list[str], typer.Option("--benchmark")],
    model_export: Annotated[Path, typer.Option("--model-export", exists=True)],
    data_root: Annotated[Path, typer.Option("--data-root", exists=True)],
    output_root: Annotated[Path, typer.Option("--output-root")] = Path(
        "evaluation-runs"
    ),
    model_name: Annotated[str, typer.Option("--model-name")] = "audio-lfm-eval",
    vllm_executable: Annotated[str, typer.Option("--vllm-executable")] = "vllm",
    config: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
    manifest: Annotated[Path, typer.Option("--manifest")] = DEFAULT_MANIFEST,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
) -> None:
    """Start one vllm serve process, run all selections, then stop it."""
    server_config, generation_config, client_config = _runtime(config)
    specs = load_specs(manifest)
    command = build_server_command(
        vllm_executable=_vllm_executable(vllm_executable),
        model_export=model_export,
        model_name=model_name,
        settings=server_config,
        chat_template_content_format="string",
        allowed_local_media_path=data_root,
    )
    server = ServerProcess(
        command=command,
        log_path=output_root / "vllm-server.log",
        environment_overrides={
            "VLLM_MAX_AUDIO_DECODE_DURATION_S": str(
                server_config.max_audio_decode_duration_seconds
            ),
            "AUDIO_LFM_AUDIO_ENCODER_MICROBATCH_SIZE": str(
                server_config.audio_encoder_microbatch_size
            ),
        },
    )
    base_url = f"http://{server_config.host}:{server_config.port}"
    with server:
        client = VllmHttpClient(
            base_url,
            timeout_seconds=float(client_config["request_timeout_seconds"]),
            max_connections=int(client_config["http_max_connections"]),
        )
        try:
            client.wait_until_ready(
                server_config.startup_timeout_seconds,
                process_check=server.ensure_running,
            )
            server.ensure_running()
            _generate_selected(
                client=client,
                selections=benchmarks,
                specs=specs,
                model_name=model_name,
                model_identity={
                    "kind": "projector-export",
                    "config_sha256": hashlib.sha256(
                        (model_export / "config.json").read_bytes()
                    ).hexdigest(),
                },
                data_root=data_root,
                output_root=output_root,
                generation=generation_config,
                client_config=client_config,
                limit=limit,
            )
        finally:
            client.close()


@app.command("run-profile-suite")
def run_profile_suite(
    profile: Annotated[str, typer.Option("--profile")],
    benchmarks: Annotated[list[str], typer.Option("--benchmark")],
    data_root: Annotated[Path, typer.Option("--data-root", exists=True)],
    output_root: Annotated[Path, typer.Option("--output-root")] = Path(
        "evaluation-runs"
    ),
    vllm_executable: Annotated[str, typer.Option("--vllm-executable")] = "vllm",
    config: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG,
    manifest: Annotated[Path, typer.Option("--manifest")] = DEFAULT_MANIFEST,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
) -> None:
    """Run selections through one persistent vLLM server for a pinned profile."""
    profiles = load_model_profiles()
    if profile not in profiles:
        raise typer.BadParameter(f"Unknown model profile {profile!r}")
    selected = profiles[profile]
    try:
        selected.require_vllm_support()
    except RuntimeError as error:
        raise typer.BadParameter(str(error)) from error
    server_config, generation_config, client_config = _runtime(config)
    specs = load_specs(manifest)
    command = build_server_command(
        vllm_executable=_vllm_executable(vllm_executable),
        model_export=selected.model_id,
        model_name=selected.name,
        settings=server_config,
        revision=selected.revision,
        load_format=None,
        allowed_local_media_path=data_root,
    )
    server = ServerProcess(
        command=command,
        log_path=output_root / selected.name / "vllm-server.log",
        plugin_name=None,
        environment_overrides={
            "VLLM_MAX_AUDIO_DECODE_DURATION_S": str(
                server_config.max_audio_decode_duration_seconds
            ),
            "AUDIO_LFM_AUDIO_ENCODER_MICROBATCH_SIZE": str(
                server_config.audio_encoder_microbatch_size
            ),
        },
    )
    base_url = f"http://{server_config.host}:{server_config.port}"
    with server:
        client = VllmHttpClient(
            base_url,
            timeout_seconds=float(client_config["request_timeout_seconds"]),
            max_connections=int(client_config["http_max_connections"]),
        )
        try:
            client.wait_until_ready(
                server_config.startup_timeout_seconds,
                process_check=server.ensure_running,
            )
            server.ensure_running()
            _generate_selected(
                client=client,
                selections=benchmarks,
                specs=specs,
                model_name=selected.name,
                model_identity={
                    "kind": "huggingface-profile",
                    "model_id": selected.model_id,
                    "revision": selected.revision,
                    "architecture": selected.architecture,
                },
                data_root=data_root,
                output_root=output_root / selected.name,
                generation=generation_config,
                client_config=client_config,
                limit=limit,
            )
        finally:
            client.close()


@app.command()
def score(
    benchmark: Annotated[str, typer.Option("--benchmark")],
    subset: Annotated[str, typer.Option("--subset")],
    output_root: Annotated[Path, typer.Option("--output-root")] = Path(
        "evaluation-runs"
    ),
    scorer_root: Annotated[Path, typer.Option("--scorer-root")] = Path(
        "evaluation-scorers"
    ),
    data_root: Annotated[Path, typer.Option("--data-root")] = Path("evaluation-data"),
    execute: Annotated[bool, typer.Option("--execute")] = False,
    manifest: Annotated[Path, typer.Option("--manifest")] = DEFAULT_MANIFEST,
) -> None:
    """Materialize official scorer input and optionally run the scorer plan."""
    spec = load_specs(manifest)[benchmark]
    run_dir = output_root / benchmark / subset
    official_input = materialize_official_input(
        spec=spec,
        subset=subset,
        predictions_path=run_dir / "predictions.jsonl",
        output_dir=run_dir / "scoring",
        data_root=data_root,
    )
    plan = scorer_plan(
        spec=spec,
        subset=subset,
        scorer_root=scorer_root,
        input_path=official_input,
        data_root=data_root,
    )
    scoring_manifest = {
        "format_version": 2,
        "benchmark": benchmark,
        "subset": subset,
        "dataset_id": spec.dataset_id,
        "dataset_revision": spec.revision,
        "scorer_repo": spec.scorer_repo,
        "scorer_revision": spec.scorer_revision,
        "official_input": str(official_input.resolve()),
        "judge": (
            {
                "provider": spec.judge_provider,
                "model": spec.judge_model,
                "credential_env": (
                    "OPENAI_API_KEY"
                    if spec.judge_provider == "openai"
                    else "OPENROUTER_API_KEY"
                ),
            }
            if subset in spec.judge_subsets
            else None
        ),
        "local_judges": (
            [
                {
                    "model": "Qwen/Qwen2.5-7B-Instruct",
                    "backend": "vllm",
                    "purpose": "open-ended rubric judge",
                },
            ]
            if benchmark == "mmau-pro"
            else []
        ),
        "unparseable_policy": "count_as_zero_in_full_denominator",
        "commands": plan,
        "security_acknowledgement_required": (
            "VoiceBench SD-QA's qa_metrics.PEDANT downloads mutable GitHub "
            "pickle files and deserializes them with joblib"
            if benchmark == "voicebench" and subset == "sd-qa"
            else None
        ),
        "omitted_official_metrics": (
            ["panda"]
            if benchmark == "voicebench" and subset == "sd-qa"
            else ["closed_ended_nvembed"]
            if benchmark == "mmau-pro"
            else []
        ),
    }
    (official_input.parent / "scoring_manifest.json").write_text(
        json.dumps(scoring_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    typer.echo("\n".join(shlex.join(command) for command in plan))
    if not execute:
        return
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str((scorer_root / benchmark).resolve()),
            str(Path(__file__).resolve().parents[1]),
        ]
    )
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    scorer_log = official_input.parent / "scorer-output.log"
    for command in plan:
        completed = subprocess.run(
            command,
            cwd=official_input.parent,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.stdout:
            typer.echo(completed.stdout, nl=False)
        if completed.stderr:
            typer.echo(completed.stderr, nl=False, err=True)
        with scorer_log.open("a", encoding="utf-8") as handle:
            handle.write(f"$ {shlex.join(command)}\n")
            handle.write(completed.stdout)
            handle.write(completed.stderr)
            handle.write(f"\n[exit_code={completed.returncode}]\n")
        completed.check_returncode()


if __name__ == "__main__":
    app()
