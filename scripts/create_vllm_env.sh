#!/usr/bin/env bash
set -euo pipefail

VENV_PATH="${VENV_PATH:-.venv-vllm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
uv venv "${VENV_PATH}" --python "${PYTHON_VERSION}" --seed
# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"
uv pip install "vllm[audio]==0.27.1" --torch-backend=auto
uv pip install -r requirements-vllm.txt
if [[ -z "${CAPTIONSTEW_REPO:-}" ]]; then
  echo "CAPTIONSTEW_REPO must point to the CaptionStew source repository" >&2
  exit 1
fi
uv pip install -e "${CAPTIONSTEW_REPO}[training]"
uv pip install -e . --no-deps
python - <<'PY'
import importlib.metadata
import torch
import transformers
import vllm
print("vLLM:", importlib.metadata.version("vllm"))
print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA runtime:", torch.version.cuda)
if importlib.metadata.version("vllm") != "0.27.1":
    raise SystemExit("Unexpected vLLM version")
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise SystemExit("CUDA with BF16 support is required")
print("GPU:", torch.cuda.get_device_name(0))
PY
