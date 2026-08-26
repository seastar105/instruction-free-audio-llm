#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv-evaluation-scorers}"
SCORER_ROOT="${SCORER_ROOT:-${REPO_ROOT}/evaluation-scorers}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

export MAX_JOBS="${MAX_JOBS:-1}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-1}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"

if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  uv venv "${VENV_PATH}" --python "${PYTHON_VERSION}" --seed
fi
uv pip install --python "${VENV_PATH}/bin/python" \
  "numpy==1.26.4" \
  "loguru==0.7.3" \
  "tqdm==4.70.0" \
  "openai==3.3.1" \
  "immutabledict==4.3.1" \
  "langdetect==1.0.9" \
  "datasets==5.0.1" \
  "nltk==3.10.3" \
  "qa-metrics==0.2.17"
uv pip install --python "${VENV_PATH}/bin/python" \
  -e "${SCORER_ROOT}/kvoicebench"

if [[ "${INSTALL_FLEXEVAL:-0}" == "1" ]]; then
  uv pip install --python "${VENV_PATH}/bin/python" \
    -e "${SCORER_ROOT}/voicebench-ja"
  # FlexEval's pyproject keeps the multilingual format-following metrics in a
  # Poetry dependency group, which editable PEP 517 installs do not include.
  uv pip install --python "${VENV_PATH}/bin/python" \
    "defusedxml>=0.7.1,<0.8" \
    "ja-sentence-segmenter==0.0.2" \
    "janome==0.5.0" \
    "spacy>=3.8.4,<4" \
    "tomli>=2.2.1,<3" \
    "pydantic[email]>=2.11.7,<3"
fi

echo "Scorer environment ready at ${VENV_PATH}"
echo "Set INSTALL_FLEXEVAL=1 to add the larger VoiceBench-JA scoring stack."
