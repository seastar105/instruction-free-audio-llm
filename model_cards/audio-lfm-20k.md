---
library_name: vllm
license: other
license_name: lfm1.0
license_link: LICENSE
language:
- en
- ko
- ja
pipeline_tag: audio-text-to-text
tags:
- audio
- multimodal
- vllm
- whisper
- lfm2.5
base_model:
- LiquidAI/LFM2.5-1.2B-Instruct
- openai/whisper-small
---

# Audio LFM 20K: Whisper-small projector for LFM2.5-1.2B

This repository contains the 20,000-update projector export from the Audio LFM
experiment. It joins a frozen `openai/whisper-small` encoder to a frozen
`LiquidAI/LFM2.5-1.2B-Instruct` decoder using a trainable 4x temporal-stacking
MLP projector. The model accepts audio as the user input and generates one text
response.

This is a lightweight projector-only vLLM export, not a standalone copy of the
1.2B decoder or Whisper. At load time the repository's `audio_lfm2` vLLM plugin
fetches the exact immutable base revisions recorded in `export_manifest.json`:

- `LiquidAI/LFM2.5-1.2B-Instruct@0f604ada3f766f9f257460c4c9f0b5d6f69d431b`
- `openai/whisper-small@973afd24965f72e36ca33b3055d56a652f456b4d`

## Model architecture

Audio is validated at 16 kHz mono, split into 30-second blocks, and every block
is padded to exactly 30 seconds for Whisper. Only effective encoder frames are
retained after each block. Four-frame stacking reduces the audio sequence to
12.5 Hz before projection into LFM's 2,048-dimensional embedding space. Long
audio and up to three ordered audio items are supported by the plugin; they are
not cropped or merged into one logical item.

The export contains ten projector tensors (about 42 MB). Whisper and LFM remain
frozen and are not included.

## Training

The projector was trained on response-expanded WavCaps and ParaSpeechCaps-Base.
For ParaSpeechCaps, each row's style caption and transcription were combined as
one offline expansion prompt to produce one response target. WavCaps used its
caption. At model training and inference time the user message contains audio
only; caption and transcript text are not supplied to the model.

Training used 16,384-token worker-local sequence packing with no example-count
cap, BF16, AdamW, peak LR `1e-3`, weight decay `1e-2`, gradient clipping at
`1.0`, 5% warmup, and cosine decay. Whisper and the LFM backbone used
`torch.compile`; the variable-length projector remained eager after a measured
compile regression. The 20K checkpoint had processed 325,464,277 LFM input
tokens, 30,712,171 supervised tokens, and 23,084,465 audio-seconds of exposure.
Its best periodic validation NLL was 0.9669.

## Evaluation

Generation used one persistent vLLM server per checkpoint with 128 concurrent
HTTP requests and the same sampling recipe as response expansion:
`temperature=0.1`, `top_k=50`, `top_p=1.0`,
`repetition_penalty=1.05`, seed 0, and `max_tokens=1024`.

| Benchmark | 6K | 20K | Metric |
| --- | ---: | ---: | --- |
| MMAU public test-mini | 22.40 | 28.20 | accuracy, % |
| MMSU | 18.38 | 21.60 | accuracy, % |
| MMAU-Pro open + instruction following | 41.67 | 47.34 | category mean, % |
| MMAR | 25.30 | 25.40 | accuracy, % |
| KMMAU | 39.02 | 39.20 | sample-weighted accuracy, % |

Selected 20K subset results:

| Suite / subset | Score |
| --- | ---: |
| VoiceBench AlpacaEval / CommonEval / WildVoice | 1.087 / 1.087 / 1.072 (1–5 judge scale) |
| VoiceBench IFEval final | 12.36% |
| VoiceBench MMSU / OpenBookQA / BBH | 0.98% / 2.20% / 0.40% |
| KVoiceBench IFEval total | 12.90% |
| KVoiceBench BBH | 11.60% |
| VoiceBench-JA M-IFEval macro / micro | 23.84% / 31.42% |
| VoiceBench-JA Elyza / Spoken-Elyza | 1.0 / 1.0 |

`evaluation-20k.json` and `evaluation-6k.json` contain every published subset,
sample count, generation parameter, judge identity, score scale, and omission.
Unparseable model responses and unparseable judge verdicts were retained in the
full denominator and scored zero.

MMAU-Pro closed-ended NV-Embed scoring was omitted by request, so its category
mean covers the open-ended and instruction-following category scores equally,
not all 5,305 rows. VoiceBench SD-QA PANDA was omitted because the upstream
dependency fetches mutable pickle files; its safe GPT metric was run and scored
0.0. These partial metrics should not be compared with a complete official
aggregate without the same scope.

## vLLM usage

Install the source repository and its isolated vLLM environment first. The
plugin is required; stock vLLM does not know the custom projector architecture.

```bash
git clone https://github.com/seastar105/instruction-free-audio-llm.git
cd instruction-free-audio-llm
bash evaluation/scripts/create_evaluation_env.sh
source .venv-evaluation/bin/activate

hf download seastar105/audio-lfm2.5-1.2b-wavcaps-paraspeech-20k \
  --local-dir model

export VLLM_PLUGINS=audio_lfm2
vllm serve model \
  --served-model-name audio-lfm-20k \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 131072 \
  --limit-mm-per-prompt '{"audio":3}' \
  --load-format safetensors \
  --enforce-eager \
  --generation-config vllm
```

Send OpenAI-compatible chat messages containing an `audio_url` content part.
For local file URLs, add an appropriately narrow `--allowed-local-media-path`.
The evaluation harness in the source repository provides resumable concurrent
clients and long-audio context preflight.

## Intended use and limitations

This checkpoint is an experimental research artifact for studying small
projectors and instruction-free audio-to-text alignment. It is not a general
ASR system, safety model, or reliable source of factual information. The frozen
decoder frequently hallucinates scene details, answers the apparent acoustic
description instead of the spoken question, or emits verbose conversational
text. Closed-question, instruction-following, Korean, and Japanese results are
especially weak. Do not use it for safety-critical, medical, legal, biometric,
surveillance, or high-stakes decisions.

Training targets inherit biases and errors from WavCaps, ParaSpeechCaps, and the
offline response generator. The model may infer sensitive speaker attributes
incorrectly. Users are responsible for respecting the source dataset licenses
and consent constraints.

## License and attribution

The projector artifact is distributed under the LFM Open License v1.0 because
it is designed to operate with LFM2.5. See `LICENSE`, including its commercial
use threshold. The Whisper dependency is Apache-2.0. Source code for the
training and vLLM plugin is separately available under MIT.
