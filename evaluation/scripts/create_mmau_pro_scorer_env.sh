#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv-evaluation-mmau-pro}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

# Keep resolver/build pressure bounded on low-memory WSL hosts. This scorer is
# intentionally separate from the API/programmatic scorer environment.
export MAX_JOBS="${MAX_JOBS:-1}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-1}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-2}"

if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  uv venv "${VENV_PATH}" --python "${PYTHON_VERSION}" --seed
fi
uv pip install --python "${VENV_PATH}/bin/python" \
  "numpy==1.26.4" \
  "pandas==2.2.3" \
  "pyarrow==16.1.0" \
  "nltk==3.9.1" \
  "scikit-learn==1.3.2" \
  "tqdm==4.66.5"

echo "MMAU-Pro scorer environment ready at ${VENV_PATH}"
echo "Run it only after vLLM generation has released the GPU."
