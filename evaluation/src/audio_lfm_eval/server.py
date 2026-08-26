from __future__ import annotations

import os
import platform
import signal
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO


@dataclass(frozen=True)
class ServerSettings:
    host: str = "127.0.0.1"
    port: int = 8000
    api_server_count: int = 1
    dtype: str = "bfloat16"
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.85
    max_num_seqs: int = 16
    max_num_batched_tokens: int = 32768
    audio_limit: int = 3
    max_audio_decode_duration_seconds: int = 720
    audio_encoder_microbatch_size: int = 4
    startup_timeout_seconds: float = 600.0


def build_server_command(
    *,
    vllm_executable: str,
    model_export: str | Path,
    model_name: str,
    settings: ServerSettings,
    revision: str | None = None,
    load_format: str | None = "safetensors",
    chat_template_content_format: str | None = None,
    allowed_local_media_path: str | Path | None = None,
) -> list[str]:
    command = [
        vllm_executable,
        "serve",
        str(model_export),
        "--served-model-name",
        model_name,
        "--host",
        settings.host,
        "--port",
        str(settings.port),
        "--api-server-count",
        str(settings.api_server_count),
        "--dtype",
        settings.dtype,
        "--tensor-parallel-size",
        "1",
        "--pipeline-parallel-size",
        "1",
        "--max-model-len",
        str(settings.max_model_len),
        "--gpu-memory-utilization",
        str(settings.gpu_memory_utilization),
        "--max-num-seqs",
        str(settings.max_num_seqs),
        "--max-num-batched-tokens",
        str(settings.max_num_batched_tokens),
        "--limit-mm-per-prompt",
        f'{{"audio":{settings.audio_limit}}}',
        "--enforce-eager",
        "--generation-config",
        "vllm",
    ]
    if revision is not None:
        command.extend(["--revision", revision])
    if load_format is not None:
        command.extend(["--load-format", load_format])
    if chat_template_content_format is not None:
        command.extend(["--chat-template-content-format", chat_template_content_format])
    if allowed_local_media_path is not None:
        command.extend(
            [
                "--allowed-local-media-path",
                str(Path(allowed_local_media_path).resolve()),
            ]
        )
    return command


class ServerProcess:
    def __init__(
        self,
        *,
        command: list[str],
        log_path: str | Path,
        plugin_name: str | None = "audio_lfm2",
        environment_overrides: Mapping[str, str] | None = None,
    ) -> None:
        self.command = command
        self.log_path = Path(log_path)
        self.plugin_name = plugin_name
        self.environment_overrides = dict(environment_overrides or {})
        self.process: subprocess.Popen[str] | None = None
        self._log_handle: IO[str] | None = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("a", encoding="utf-8")
        environment = os.environ.copy()
        environment.update(
            {
                "MAX_JOBS": "1",
                "CMAKE_BUILD_PARALLEL_LEVEL": "1",
                "TORCHINDUCTOR_COMPILE_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
        )
        environment.update(self.environment_overrides)
        if "microsoft" in platform.release().lower():
            # CUDA UVA is unavailable under WSL, so vLLM's V2 runner cannot
            # allocate its host-side staged-write buffer. FlashInfer's sampler
            # also attempts a CUDA JIT build through /usr/local/cuda/nvcc.
            environment.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
            environment.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        if self.plugin_name is not None:
            environment["VLLM_PLUGINS"] = self.plugin_name
        self.process = subprocess.Popen(
            self.command,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
            start_new_session=True,
        )

    def ensure_running(self) -> None:
        if self.process is None:
            raise RuntimeError("vLLM server process has not started")
        code = self.process.poll()
        if code is not None:
            raise RuntimeError(
                f"vLLM server exited with code {code}; inspect {self.log_path}"
            )

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
        if self._log_handle is not None:
            self._log_handle.close()

    def __enter__(self) -> ServerProcess:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
