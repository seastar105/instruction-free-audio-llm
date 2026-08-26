#!/usr/bin/env bash
set -euo pipefail
uv run --no-sync audio-lfm preflight --config configs/paraspeech_whisper_lfm2_smoke.yaml
uv run --no-sync audio-lfm train --config configs/paraspeech_whisper_lfm2_smoke.yaml
