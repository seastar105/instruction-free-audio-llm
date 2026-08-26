#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv-evaluation}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

export MAX_JOBS="${MAX_JOBS:-1}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"

uv venv "${VENV_PATH}" --python "${PYTHON_VERSION}" --seed
VIRTUAL_ENV="${VENV_PATH}" uv pip install \
  --python "${VENV_PATH}/bin/python" \
  "vllm[audio]==0.27.1" \
  --torch-backend=auto
VIRTUAL_ENV="${VENV_PATH}" uv sync \
  --project "${REPO_ROOT}/evaluation" \
  --python "${VENV_PATH}/bin/python" \
  --extra dev \
  --inexact \
  --active
uv pip install \
  --python "${VENV_PATH}/bin/python" \
  --no-deps \
  -e "${REPO_ROOT}"

"${VENV_PATH}/bin/python" - <<'PY'
import importlib.metadata

assert importlib.metadata.version("vllm") == "0.27.1"
assert importlib.metadata.version("audio-lfm-eval") == "0.1.0"
assert importlib.metadata.version("audio-lfm") == "0.1.0"
print("evaluation environment ready")
PY
