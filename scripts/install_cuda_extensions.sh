#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import platform
import sys

import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA PyTorch must be installed first")
print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("BF16:", torch.cuda.is_bf16_supported())
if not torch.cuda.is_bf16_supported():
    raise SystemExit("This training configuration requires CUDA BF16 support")
if sys.version_info[:2] != (3, 12):
    raise SystemExit("The prebuilt FlashAttention wheel requires CPython 3.12")
if platform.machine() != "x86_64":
    raise SystemExit("The pinned FlashAttention wheel requires Linux x86_64")
if not torch.__version__.startswith("2.10.") or torch.version.cuda != "13.0":
    raise SystemExit("The pinned FlashAttention wheel requires Torch 2.10 + CUDA 13.0")
if not torch.compiled_with_cxx11_abi():
    raise SystemExit("The pinned FlashAttention wheel requires CXX11 ABI")
PY

uv pip install --no-deps \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
uv pip install --no-build-isolation "causal-conv1d==1.6.2.post1"
