# No-System-Prompt Training Correction

This patch corrects the caption-expansion and projector-training prompt relationship.

The expansion-only system message is:

```text
You are an AI assistant directly hearing this audio. Respond as if you heard it yourself.
```

It must exist **only while converting an official caption into an expanded response**.

It must not appear in:

* projector training;
* Hugging Face teacher-forced audio evaluation;
* vLLM audio generation evaluation;
* the audio-model export;
* the audio prompt file;
* the audio prompt manifest;
* qualitative audio inference;
* dMel training or evaluation.

---

# 1. Exact prompt separation

## 1.1 Caption expansion

Caption expansion uses exactly:

```python
CAPTION_EXPANSION_SYSTEM_PROMPT = (
    "You are an AI assistant directly hearing this audio. "
    "Respond as if you heard it yourself."
)


def build_caption_expansion_messages(
    caption: str,
) -> list[dict[str, str]]:
    if not isinstance(caption, str):
        raise TypeError("caption must be a string")

    if not caption.strip():
        raise ValueError("caption must not be empty")

    # Preserve the official caption exactly as cataloged.
    return [
        {
            "role": "system",
            "content": CAPTION_EXPANSION_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": caption,
        },
    ]
```

This stage produces:

```text
expanded_response
```

The expansion job does not consume audio.

## 1.2 Projector training

Projector training uses exactly one input message:

```python
def build_audio_prompt_messages(
    audio_sentinel: str,
) -> list[dict[str, str]]:
    if not audio_sentinel:
        raise ValueError("audio_sentinel must not be empty")

    return [
        {
            "role": "user",
            "content": audio_sentinel,
        },
    ]
```

The full supervised conversation is:

```python
def build_audio_training_messages(
    *,
    audio_sentinel: str,
    expanded_response: str,
) -> list[dict[str, str]]:
    if not expanded_response.strip():
        raise ValueError("expanded_response must not be empty")

    return [
        {
            "role": "user",
            "content": audio_sentinel,
        },
        {
            "role": "assistant",
            "content": expanded_response,
        },
    ]
```

There is no:

```python
{
    "role": "system",
    ...
}
```

in either training function.

Do not insert the expansion instruction as:

* a system message;
* a user-message prefix;
* plain text before the audio;
* a learned prompt;
* a tokenizer prefix;
* a hidden configuration default.

The tokenizer’s unchanged chat template may emit its normal role-control tokens. The code must not inject any additional system content.

---

# 2. Correct semantic relationship

The preprocessing and training contexts are intentionally different.

## Expansion

```text
system:
    You are an AI assistant directly hearing this audio.
    Respond as if you heard it yourself.

user:
    <official style caption>

assistant:
    <generated expanded response>
```

## Training

```text
user:
    <projected audio embeddings>

assistant:
    <generated expanded response>
```

The objective is not prompt-context reproduction.

The objective is to make projected audio sufficient to elicit the response that the frozen decoder generated from the caption-based semantic surrogate.

Conceptually:

```text
official style caption
    -> expansion-only instruction + caption
    -> frozen decoder
    -> expanded response

audio
    -> frozen audio encoder
    -> trainable projector
    -> frozen decoder, without expansion instruction
    -> same expanded response
```

The source caption and expansion system prompt are both absent from the projector-training input.

---

# 3. Replace the previous Section 23.1

Use this replacement.

## 23.1 Relationship between expansion and audio training

Caption expansion and projector training share:

* decoder model ID;
* immutable decoder checkpoint revision;
* tokenizer revision;
* tokenizer chat template;
* tokenizer control-token semantics.

They do not share the same message list.

Caption expansion uses:

```python
[
    {
        "role": "system",
        "content": (
            "You are an AI assistant directly hearing this audio. "
            "Respond as if you heard it yourself."
        ),
    },
    {
        "role": "user",
        "content": official_caption,
    },
]
```

Projector training uses:

```python
[
    {
        "role": "user",
        "content": AUDIO_SENTINEL,
    },
]
```

The training target is:

```text
the assistant response generated during caption expansion
```

The caption text and expansion system instruction are preprocessing inputs only.

They must never become projector-training inputs.

---

# 4. Correct decoder-lock behavior

The decoder lock must prove model identity and tokenizer-format compatibility.

It must not require expansion-prompt and training-prompt equality.

Use:

```json
{
  "format_version": 2,

  "model_id": "LiquidAI/LFM2.5-1.2B-Instruct",
  "model_revision": "<immutable commit SHA>",

  "tokenizer_id": "LiquidAI/LFM2.5-1.2B-Instruct",
  "tokenizer_revision": "<immutable commit SHA>",
  "chat_template_sha256": "<sha256>",

  "expansion_system_prompt_sha256": "<sha256>",
  "expansion_prompt_template_sha256": "<sha256>",
  "generation_config_sha256": "<sha256>",
  "expansion_recipe_sha256": "<sha256>"
}
```

At projector-training startup, require:

```text
training decoder model ID
    ==
expansion decoder model ID

training decoder immutable revision
    ==
expansion decoder immutable revision

training tokenizer revision
    ==
expansion tokenizer revision

training tokenizer chat-template SHA
    ==
expansion tokenizer chat-template SHA
```

Do not require:

```text
training prompt-template SHA
    ==
expansion prompt-template SHA
```

Do not require:

```text
training system-prompt SHA
    ==
expansion system-prompt SHA
```

The prompts are supposed to differ.

The training run manifest must store a separate:

```text
audio_training_prompt_template_sha256
```

The expansion manifest continues to store:

```text
expansion_prompt_template_sha256
expansion_system_prompt_sha256
```

The two hashes should normally be different.

Add an assertion:

```python
if training_prompt_template_sha256 == expansion_prompt_template_sha256:
    raise ConfigurationError(
        "Expansion and audio-training prompts unexpectedly match. "
        "The expansion system instruction must not be used during training."
    )
```

This comparison may be disabled only for unrelated future prompt modes. It is mandatory for:

```text
caption_expansion_alignment
```

---

# 5. Correct training configuration

Replace the previous expanded-target training prompt configuration with:

```yaml
targets:
  provider: caption_expansion_overlay

  overlay_kind: response
  overlay_source: caption_expansion_v1
  response_type: audio_assistant_response

  expansion_decoder_lock: >-
    runs/caption-expansion/train_base/decoder_lock.json

  expansion_recipe_sha256: "<expected expansion recipe SHA>"

  source_target_type: style_caption

  review_status_allowlist:
    - unreviewed
    - accepted

  missing_policy: error
  duplicate_policy: error

  selection:
    train: one_per_audio_per_epoch
    validation: all
    final_evaluation: all

prompt:
  mode: caption_expansion_alignment

  # No system message is permitted for this mode.
  system_message: null
  require_no_system_message: true

  user_content: audio_sentinel_only
  audio_sentinel: "<<__AUDIO_EMBEDDINGS_08E8F7E7__>>"

  supervise_assistant_termination: true
```

Strict configuration validation must reject:

```yaml
prompt:
  mode: caption_expansion_alignment
  system_message: "anything"
```

It must also reject an omitted `system_message` field if the default would be non-null.

For this prompt mode, normalize the resolved configuration to:

```yaml
system_message: null
```

---

# 6. Revised prompt compiler

Use the tokenizer chat template without a system message.

## 6.1 Prompt-only rendering

```python
prompt_messages = [
    {
        "role": "user",
        "content": audio_sentinel,
    },
]

prompt_only_text = tokenizer.apply_chat_template(
    prompt_messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

## 6.2 Full supervised rendering

```python
full_messages = [
    {
        "role": "user",
        "content": audio_sentinel,
    },
    {
        "role": "assistant",
        "content": expanded_response,
    },
]

full_text = tokenizer.apply_chat_template(
    full_messages,
    tokenize=False,
    add_generation_prompt=False,
)
```

## 6.3 Required assertions

```python
if any(message["role"] == "system" for message in prompt_messages):
    raise PromptCompilationError("System messages are forbidden during audio training")

if any(message["role"] == "system" for message in full_messages):
    raise PromptCompilationError("System messages are forbidden during audio training")

if CAPTION_EXPANSION_SYSTEM_PROMPT in prompt_only_text:
    raise PromptCompilationError("Expansion system prompt leaked into training prompt")

if CAPTION_EXPANSION_SYSTEM_PROMPT in full_text:
    raise PromptCompilationError(
        "Expansion system prompt leaked into supervised training text"
    )

if prompt_only_text.count(audio_sentinel) != 1:
    raise PromptCompilationError("Audio sentinel must occur exactly once")

if full_text.count(audio_sentinel) != 1:
    raise PromptCompilationError("Audio sentinel must occur exactly once")
```

Do not attempt to remove or rewrite model-specific chat-template control tokens.

Only prohibit explicit application-provided system content.

---

# 7. Revised sequence construction

The logical sequence is now:

```text
user-role prefix tokens
audio-start vector
projected audio vectors
audio-end vector
user-role suffix tokens
assistant-generation prefix tokens
expanded-response tokens
assistant termination tokens
```

It does not contain:

```text
system-role tokens
expansion instruction tokens
official caption tokens
transcription tokens
```

Labels are:

```text
-100
```

for:

* user-role tokens;
* audio-start vector;
* projected audio vectors;
* audio-end vector;
* assistant-generation-prefix tokens.

Labels contain actual token IDs only for:

* expanded-response tokens;
* configured assistant termination tokens.

---

# 8. Correct prompt-splitting algorithm

Use the existing sentinel-splitting mechanism with the corrected messages.

```python
def compile_expanded_audio_target(
    *,
    tokenizer,
    audio_sentinel: str,
    expanded_response: str,
) -> PreparedText:
    prompt_messages = build_audio_prompt_messages(
        audio_sentinel,
    )

    full_messages = build_audio_training_messages(
        audio_sentinel=audio_sentinel,
        expanded_response=expanded_response,
    )

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    if CAPTION_EXPANSION_SYSTEM_PROMPT in prompt_text:
        raise PromptCompilationError(
            "Expansion instruction leaked into training prompt"
        )

    if CAPTION_EXPANSION_SYSTEM_PROMPT in full_text:
        raise PromptCompilationError(
            "Expansion instruction leaked into training target sequence"
        )

    if prompt_text.count(audio_sentinel) != 1:
        raise PromptCompilationError("Prompt must contain exactly one audio sentinel")

    if full_text.count(audio_sentinel) != 1:
        raise PromptCompilationError(
            "Full conversation must contain exactly one audio sentinel"
        )

    before_audio_text, after_audio_prompt_text = prompt_text.split(audio_sentinel)

    full_before_audio_text, full_after_audio_text = full_text.split(audio_sentinel)

    if before_audio_text != full_before_audio_text:
        raise PromptCompilationError("Prompt/full text disagree before audio position")

    if not full_after_audio_text.startswith(after_audio_prompt_text):
        raise PromptCompilationError("Assistant target boundary cannot be derived")

    target_suffix_text = full_after_audio_text[len(after_audio_prompt_text) :]

    before_audio_ids = tokenizer(
        before_audio_text,
        add_special_tokens=False,
    ).input_ids

    after_audio_prompt_ids = tokenizer(
        after_audio_prompt_text,
        add_special_tokens=False,
    ).input_ids

    target_suffix_ids = tokenizer(
        target_suffix_text,
        add_special_tokens=False,
    ).input_ids

    combined_after_ids = tokenizer(
        after_audio_prompt_text + target_suffix_text,
        add_special_tokens=False,
    ).input_ids

    if combined_after_ids != (after_audio_prompt_ids + target_suffix_ids):
        raise PromptCompilationError(
            "Tokenizer merge occurred at assistant-target boundary; "
            "use fast-tokenizer offset mapping"
        )

    return PreparedText(
        before_audio_ids=tuple(before_audio_ids),
        after_audio_prompt_ids=tuple(after_audio_prompt_ids),
        target_suffix_ids=tuple(target_suffix_ids),
        target_id=...,
        prompt_sha256=...,
    )
```

The omitted fields should be populated from the selected expansion record.

---

# 9. Expansion manifest versus training manifest

Keep prompt provenance separate.

## Expansion manifest

Records:

```text
expansion system prompt
expansion message template
caption-derived rendered prompt
generation configuration
decoder identity
chat-template identity
```

## Training manifest

Records:

```text
no-system training message template
audio sentinel
projector configuration
decoder identity
chat-template identity
selected expansion recipe
```

Recommended fields:

```json
{
  "expansion_recipe_sha256": "<sha256>",

  "expansion_system_prompt_sha256": "<sha256>",
  "expansion_prompt_template_sha256": "<sha256>",

  "audio_training_has_system_message": false,
  "audio_training_prompt_template_sha256": "<sha256>",

  "decoder_model_id": "LiquidAI/LFM2.5-1.2B-Instruct",
  "decoder_revision": "<immutable SHA>",
  "tokenizer_revision": "<immutable SHA>",
  "chat_template_sha256": "<sha256>"
}
```

Hard assertion:

```python
if manifest["audio_training_has_system_message"] is not False:
    raise ManifestError("Expanded-target audio training must not use a system message")
```

---

# 10. Hugging Face evaluation correction

Teacher-forced evaluation must use the same no-system audio prompt as training.

For every audio item:

```python
messages = [
    {
        "role": "user",
        "content": AUDIO_SENTINEL,
    },
]
```

For every expanded reference:

```python
full_messages = [
    {
        "role": "user",
        "content": AUDIO_SENTINEL,
    },
    {
        "role": "assistant",
        "content": expanded_reference,
    },
]
```

Do not add the caption-expansion system instruction during:

* development NLL;
* holdout NLL;
* generated-response inspection;
* first-token parity evaluation.

The source expansion records retain their system-prompt hash for provenance, but that prompt is not reconstructed for audio evaluation.

---

# 11. vLLM audio-evaluation correction

The custom audio-vLLM plugin continues to use one audio placeholder, but the rendered request contains no system message.

Build the request prompt from:

```python
audio_eval_messages = [
    {
        "role": "user",
        "content": "<|audio|>",
    },
]

rendered_prompt = tokenizer.apply_chat_template(
    audio_eval_messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

Or pretokenize:

```python
prompt_token_ids = tokenizer.apply_chat_template(
    audio_eval_messages,
    tokenize=True,
    add_generation_prompt=True,
)
```

Do not use:

```python
[
    {
        "role": "system",
        "content": CAPTION_EXPANSION_SYSTEM_PROMPT,
    },
    {
        "role": "user",
        "content": "<|audio|>",
    },
]
```

The vLLM export must store:

```json
{
  "audio_inference_has_system_message": false,
  "audio_inference_prompt_template_sha256": "<sha256>",
  "expansion_recipe_sha256": "<sha256>"
}
```

The expansion recipe hash identifies how target responses were created. It does not define the inference prompt.

Update vLLM prompt parity tests so that both HF and vLLM use:

```python
[
    {
        "role": "user",
        "content": AUDIO_PLACEHOLDER,
    },
]
```

---

# 12. vLLM request-builder assertion

Add:

```python
def build_vllm_audio_request_prompt(
    *,
    tokenizer,
    audio_token: str,
) -> tuple[str, list[int]]:
    messages = [
        {
            "role": "user",
            "content": audio_token,
        },
    ]

    if any(message["role"] == "system" for message in messages):
        raise ValueError("System message is forbidden for audio inference")

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )

    if CAPTION_EXPANSION_SYSTEM_PROMPT in rendered:
        raise ValueError("Caption-expansion instruction leaked into audio inference")

    if rendered.count(audio_token) != 1:
        raise ValueError("Audio placeholder must occur exactly once")

    return rendered, list(token_ids)
```

---

# 13. Configuration validation

Implement a mode-specific validator.

```python
@model_validator(mode="after")
def validate_prompt_mode(self):
    if self.mode == "caption_expansion_alignment":
        if self.system_message is not None:
            raise ValueError(
                "caption_expansion_alignment forbids a training-time system message"
            )

        if self.user_content != "audio_sentinel_only":
            raise ValueError("caption_expansion_alignment requires audio_sentinel_only")

    return self
```

For vLLM evaluation:

```python
@model_validator(mode="after")
def validate_audio_eval_prompt(self):
    if self.system_message is not None:
        raise ValueError("Audio evaluation must not use a system message")

    return self
```

The caption-expansion configuration remains the only configuration containing:

```yaml
prompt:
  system_message: >-
    You are an AI assistant directly hearing this audio. Respond as if you
    heard it yourself.
```

---

# 14. Remove these earlier requirements

Delete or replace every occurrence of:

```text
Caption expansion and audio training use the same system message.
```

Delete:

```text
training system prompt SHA
    ==
expansion system prompt SHA
```

Delete this training configuration:

```yaml
prompt:
  system_message: >-
    You are an AI assistant directly hearing this audio. Respond as if you
    heard it yourself.
```

Delete this training message:

```python
{
    "role": "system",
    "content": CAPTION_EXPANSION_SYSTEM_PROMPT,
}
```

Delete any acceptance criterion stating:

```text
The training system message matches expansion exactly.
```

Replace it with:

```text
The expansion system message is absent from training and audio evaluation.
```

---

# 15. Revised tests

## 15.1 Expansion prompt test

Assert exact equality:

```python
assert build_caption_expansion_messages(caption) == [
    {
        "role": "system",
        "content": (
            "You are an AI assistant directly hearing this audio. "
            "Respond as if you heard it yourself."
        ),
    },
    {
        "role": "user",
        "content": caption,
    },
]
```

## 15.2 Training prompt-role test

```python
messages = build_audio_prompt_messages(AUDIO_SENTINEL)

assert messages == [
    {
        "role": "user",
        "content": AUDIO_SENTINEL,
    },
]

assert all(message["role"] != "system" for message in messages)
```

## 15.3 Full training-message test

```python
messages = build_audio_training_messages(
    audio_sentinel=AUDIO_SENTINEL,
    expanded_response="The speaker sounds calm.",
)

assert [message["role"] for message in messages] == [
    "user",
    "assistant",
]
```

## 15.4 Rendered-text leakage test

```python
prompt_text = tokenizer.apply_chat_template(
    build_audio_prompt_messages(AUDIO_SENTINEL),
    tokenize=False,
    add_generation_prompt=True,
)

assert CAPTION_EXPANSION_SYSTEM_PROMPT not in prompt_text
```

Repeat for the full supervised rendering.

## 15.5 Caption leakage test

For a known source caption:

```python
assert source_caption not in prompt_text
assert source_caption not in model_input_debug_text
```

This test should use a distinctive synthetic caption unlikely to appear in control tokens.

## 15.6 Manifest mismatch test

Training must accept:

```text
same model revision
same tokenizer revision
same chat-template hash
different prompt-template hash
```

Training must reject:

```text
different model revision
different tokenizer revision
different chat-template hash
```

This verifies that prompt equality is no longer incorrectly enforced.

## 15.7 HF/vLLM prompt parity test

HF and vLLM must render or tokenize the same no-system message:

```python
[
    {
        "role": "user",
        "content": AUDIO_PLACEHOLDER,
    },
]
```

Compare token IDs before and after replacing the placeholder with multimodal embeddings.

## 15.8 System-prompt exclusivity test

Search the resolved run artifacts.

The expansion system string may exist only in:

```text
caption-expansion configuration
caption-expansion decoder lock
caption-expansion manifest
caption-expansion local records
caption-expansion overlay provenance
```

It must not exist in:

```text
training resolved configuration
training prompt file
training prompt examples
training model inputs
HF audio-evaluation prompt
vLLM audio-evaluation prompt
vLLM export inference prompt
generation prediction prompt
```

Do not search model weights or tokenizer assets as text.

---

# 16. Revised acceptance criteria

Replace the relevant acceptance criteria with:

1. Caption expansion uses the exact supplied system and user messages.
2. The system message exists only during caption expansion.
3. Projector training contains no explicit system message.
4. HF audio evaluation contains no explicit system message.
5. vLLM audio evaluation contains no explicit system message.
6. The official caption is used only as the expansion-stage user content.
7. The official caption is absent from projector-training input.
8. The expanded response is the projector-training assistant target.
9. Expansion and training use the same immutable decoder checkpoint.
10. Expansion and training use the same tokenizer revision.
11. Expansion and training use the same tokenizer chat template.
12. Expansion and training are not required to use the same prompt template.
13. Expansion-prompt and training-prompt hashes are stored separately.
14. Training fails if an application-provided system message is configured.
15. The no-system training prompt is shared by HF and vLLM audio evaluation.
16. The caption-expansion recipe remains attached to targets as provenance.
17. No code path silently restores the expansion instruction at inference.
18. Direct-caption and expanded-response experiments remain separate configurations.

---

# 17. Final canonical examples

## Caption expansion

```python
messages = [
    {
        "role": "system",
        "content": (
            "You are an AI assistant directly hearing this audio. "
            "Respond as if you heard it yourself."
        ),
    },
    {
        "role": "user",
        "content": caption,
    },
]
```

## Projector training prompt

```python
messages = [
    {
        "role": "user",
        "content": AUDIO_SENTINEL,
    },
]
```

## Projector training full target

```python
messages = [
    {
        "role": "user",
        "content": AUDIO_SENTINEL,
    },
    {
        "role": "assistant",
        "content": expanded_response,
    },
]
```

## vLLM audio evaluation

```python
messages = [
    {
        "role": "user",
        "content": "<|audio|>",
    },
]
```

The expansion-only instruction must not appear in the final three cases.
