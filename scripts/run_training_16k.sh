#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PATH="${project_root}/.venv/bin:${PATH}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${project_root}/.runtime/hf-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${project_root}/.runtime/torchinductor-cache}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export CAPTIONSTEW_ROOT="${CAPTIONSTEW_ROOT:-${project_root}/.runtime/captionstew/CaptionStew}"
export CAPTION_EXPANSION_DECODER_LOCK="${CAPTION_EXPANSION_DECODER_LOCK:-${project_root}/.runtime/captionstew/decoder_lock.json}"
export CAPTION_EXPANSION_RECIPE_SHA256="${CAPTION_EXPANSION_RECIPE_SHA256:-9155f96b7c98746b83ad9a20638ee1ed29831d991d755d5e80ff560f86c317e4}"
export WAVCAPS_EXPANSION_DECODER_LOCK="${WAVCAPS_EXPANSION_DECODER_LOCK:-${project_root}/.runtime/captionstew/wavcaps_decoder_lock.json}"
export WAVCAPS_EXPANSION_RECIPE_SHA256="${WAVCAPS_EXPANSION_RECIPE_SHA256:-9b1594c781ccdc977191f74030b8d3fd7ee3347fb207529d1ca411540fe04638}"

cd "${project_root}"
config_path="${1:-configs/wavcaps_paraspeech_whisper_small_lfm2_expanded_3epoch.yaml}"
exec .venv/bin/audio-lfm train \
  --config "${config_path}"
