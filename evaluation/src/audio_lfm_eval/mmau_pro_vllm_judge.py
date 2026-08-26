from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


def completed_result_count(output_path: Path, expected: int) -> int | None:
    """Return the record count only when a prior judge output is complete."""
    if not output_path.is_file():
        return None
    try:
        records = [
            json.loads(line)
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return None
    if len(records) != expected:
        return None
    for index, record in enumerate(records):
        if record.get("open_index") != index or "evaluation" not in record:
            return None
    return len(records)


def create_evaluation_prompt(
    question: str,
    reference_answer: str,
    model_response: str,
) -> str:
    return (
        "You are an expert evaluator for general question answering tasks. "
        "Please evaluate the quality of a model's response to a question.\n\n"
        f"Question: {question}\n\n"
        f"Reference Answer: {reference_answer}\n\n"
        f"Model Response: {model_response}\n\n"
        "Please evaluate the model response on the following criteria and "
        "provide scores from 1-5 (where 5 is best):\n\n"
        "1. **Correctness**: How factually accurate is the response compared "
        "to the reference?\n"
        "2. **Relevance**: How well does the response address the specific "
        "question asked?\n"
        "3. **Completeness**: Does the response cover all important aspects "
        "mentioned in the reference?\n"
        "4. **Clarity**: How clear and well-structured is the response?\n\n"
        "For each criterion, provide:\n"
        "- A score from 1-5\n"
        "- A brief justification (1-2 sentences)\n\n"
        "Format your response as:\n\n"
        "CORRECTNESS: [score] - [justification]\n"
        "RELEVANCE: [score] - [justification]\n"
        "COMPLETENESS: [score] - [justification]\n"
        "CLARITY: [score] - [justification]\n"
        "OVERALL: [average score] - [overall assessment]"
    )


def build_conversations(records: list[dict[str, Any]]) -> list[list[dict[str, str]]]:
    conversations: list[list[dict[str, str]]] = []
    for record in records:
        conversations.append(
            [
                {
                    "role": "system",
                    "content": "You are a helpful and objective evaluator.",
                },
                {
                    "role": "user",
                    "content": create_evaluation_prompt(
                        str(record["question"]),
                        str(record["answer"]),
                        str(record.get("model_output") or ""),
                    ),
                },
            ]
        )
    return conversations


def run_judge(
    input_path: Path,
    output_path: Path,
    *,
    model_name: str = MODEL_NAME,
    force: bool = False,
) -> None:
    table = pq.read_table(input_path)
    records = [
        record for record in table.to_pylist() if record.get("category") == "open"
    ]
    if not force and completed_result_count(output_path, len(records)) is not None:
        print(f"Reusing {len(records)} complete vLLM judge results: {output_path}")
        return

    # vLLM's V2 runner requires UVA, which is not exposed by every WSL CUDA
    # driver. The V1 runner remains a fully batched vLLM backend and works in
    # both native Linux and WSL environments.
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    # The optional FlashInfer sampler JIT-compiles when its matching prebuilt
    # kernel is absent. This environment has a CUDA runtime but no nvcc, so use
    # vLLM's native CUDA/PyTorch top-k/top-p sampler instead.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams

    conversations = build_conversations(records)

    llm = LLM(
        model=model_name,
        runner="generate",
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.88,
        max_num_seqs=64,
        max_num_batched_tokens=32768,
        enable_prefix_caching=True,
        # Only 625 prompts are judged per checkpoint. Eager execution avoids
        # several minutes of CUDA-graph compilation on WSL while preserving
        # vLLM's continuous batching and PagedAttention scheduler.
        enforce_eager=True,
        seed=0,
    )
    sampling = SamplingParams(
        temperature=0.1,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.05,
        max_tokens=512,
        seed=0,
    )
    outputs = llm.chat(
        cast(Any, conversations),
        sampling_params=sampling,
        use_tqdm=True,
    )
    if len(outputs) != len(records):
        raise RuntimeError(
            f"vLLM returned {len(outputs)} outputs for {len(records)} prompts"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, output in enumerate(outputs):
            text = output.outputs[0].text if output.outputs else ""
            handle.write(
                json.dumps(
                    {"open_index": index, "evaluation": text},
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch MMAU-Pro's local Qwen judge with vLLM."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_judge(args.input, args.output, model_name=args.model, force=args.force)


if __name__ == "__main__":
    main()
