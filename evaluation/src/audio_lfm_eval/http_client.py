from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class GenerationSettings:
    max_tokens: int = 1024
    temperature: float = 0.1
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.05
    seed: int = 0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class VllmHttpClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 600.0,
        max_connections: int = 256,
    ) -> None:
        if max_connections <= 0:
            raise ValueError("max_connections must be positive")
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(
            timeout=timeout_seconds,
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
            ),
        )

    def close(self) -> None:
        self.client.close()

    def health(self) -> bool:
        try:
            return self.client.get(f"{self.base_url}/health").status_code == 200
        except httpx.HTTPError:
            return False

    def wait_until_ready(
        self,
        timeout_seconds: float,
        process_check: Callable[[], None] | None = None,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if process_check is not None:
                process_check()
            if self.health():
                return
            time.sleep(1.0)
        raise TimeoutError(f"vLLM server did not become healthy at {self.base_url}")

    def generate(
        self,
        *,
        model: str,
        audio_urls: list[str],
        question: str,
        choices: tuple[str, ...],
        settings: GenerationSettings,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [
            {"type": "audio_url", "audio_url": {"url": url}} for url in audio_urls
        ]
        prompt = question.strip()
        if choices:
            rendered = "\n".join(
                f"{chr(65 + index)}. {choice}" for index, choice in enumerate(choices)
            )
            prompt = (
                f"{prompt}\n\nChoices:\n{rendered}\n\n"
                "Respond with the choice letter followed by the choice text."
            )
        if prompt:
            content.append({"type": "text", "text": prompt})
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": settings.max_tokens,
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "seed": settings.seed,
            "top_k": settings.top_k,
            "repetition_penalty": settings.repetition_penalty,
        }
        response = self.client.post(
            f"{self.base_url}/v1/chat/completions", json=payload
        )
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][0]
        return {
            "prediction": choice["message"].get("content", ""),
            "finish_reason": choice.get("finish_reason"),
            "usage": body.get("usage", {}),
            "response_id": body.get("id"),
        }
