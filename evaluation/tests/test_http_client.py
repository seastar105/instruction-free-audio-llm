import json

import httpx
import pytest

from audio_lfm_eval.http_client import GenerationSettings, VllmHttpClient


def test_http_client_rejects_nonpositive_connection_limit() -> None:
    with pytest.raises(ValueError, match="max_connections must be positive"):
        VllmHttpClient("http://vllm.invalid", max_connections=0)


def test_qwen_smoke_uses_vllm_openai_audio_url_schema() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "smoke",
                "choices": [
                    {"message": {"content": "answer"}, "finish_reason": "stop"}
                ],
                "usage": {},
            },
        )

    client = VllmHttpClient("http://vllm.invalid")
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        client.generate(
            model="qwen25-omni-3b-smoke",
            audio_urls=["data:audio/wav;base64,UklGRg=="],
            question="What is audible?",
            choices=(),
            settings=GenerationSettings(max_tokens=1024),
        )
    finally:
        client.close()

    assert captured["model"] == "qwen25-omni-3b-smoke"
    assert captured["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {"url": "data:audio/wav;base64,UklGRg=="},
                },
                {"type": "text", "text": "What is audible?"},
            ],
        }
    ]
    assert captured["max_tokens"] == 1024
    assert captured["temperature"] == 0.1
    assert captured["top_p"] == 1.0
    assert captured["top_k"] == 50
    assert captured["repetition_penalty"] == 1.05
    assert captured["seed"] == 0


def test_multiple_audio_items_preserve_their_order() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "multi",
                "choices": [{"message": {"content": "A"}, "finish_reason": "stop"}],
                "usage": {},
            },
        )

    client = VllmHttpClient("http://vllm.invalid")
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        client.generate(
            model="audio-lfm-eval",
            audio_urls=["data:first", "data:second", "data:third"],
            question="Compare the clips.",
            choices=(),
            settings=GenerationSettings(),
        )
    finally:
        client.close()

    content = captured["messages"][0]["content"]  # type: ignore[index]
    assert [item["audio_url"]["url"] for item in content[:-1]] == [
        "data:first",
        "data:second",
        "data:third",
    ]
