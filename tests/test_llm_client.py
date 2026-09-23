from __future__ import annotations

import json

import httpx
import pytest

from jev_rank.llm_client import (
    ProviderClient,
    ProviderResponseError,
    build_provider_payload,
    parse_gemini_response,
    parse_openai_response,
)


def test_openai_binary_response_parsing():
    body = {
        "status": "completed",
        "model": "gpt-5.4-nano-2026-03-17",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps({"relevant": True, "relevance_probability": 0.91})}]}],
        "usage": {"input_tokens": 120, "output_tokens": 12},
    }
    parsed = parse_openai_response(body, "openai-binary")
    assert parsed.score == 0.91
    assert parsed.input_tokens == 120
    assert parsed.resolved_model == "gpt-5.4-nano-2026-03-17"


def test_gemini_graded_response_parsing_includes_thinking_tokens():
    body = {
        "modelVersion": "gemini-3.5-flash-lite-001",
        "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps({"label": "Directly useful", "relevance_score": 3.0})}]}}],
        "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 8, "thoughtsTokenCount": 4},
    }
    parsed = parse_gemini_response(body, "gemini-graded")
    assert parsed.score == 3.0
    assert parsed.output_tokens == 12


@pytest.mark.parametrize(
    "body,method,parser",
    [
        ({"status": "completed", "model": "m", "output": [], "usage": {"input_tokens": 1, "output_tokens": 1}}, "openai-binary", parse_openai_response),
        ({"modelVersion": "m", "candidates": [], "usageMetadata": {"promptTokenCount": 1}}, "gemini-graded", parse_gemini_response),
    ],
)
def test_provider_malformed_responses(body, method, parser):
    with pytest.raises(ProviderResponseError):
        parser(body, method)


def test_payloads_use_fixed_models_and_structured_schemas():
    openai = build_provider_payload("openai-binary", "q", "d", "gpt-model")
    assert openai["model"] == "gpt-model"
    assert openai["reasoning"] == {"effort": "none"}
    assert openai["text"]["format"]["strict"] is True
    gemini = build_provider_payload("gemini-graded", "q", "d", "gemini-model")
    assert gemini["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "minimal"}
    assert gemini["generationConfig"]["responseMimeType"] == "application/json"


@pytest.mark.asyncio
async def test_provider_retry_429():
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "model": "gpt-model-001",
                "output_text": json.dumps({"relevant": False, "relevance_probability": 0.1}),
                "usage": {"input_tokens": 10, "output_tokens": 4},
            },
        )

    async def no_sleep(_: float):
        return None

    payload = build_provider_payload("openai-binary", "q", "d", "gpt-model")
    async with ProviderClient("openai", "secret", "gpt-model", max_retries=2, transport=httpx.MockTransport(handler), sleep=no_sleep) as client:
        call = await client.score(payload, "openai-binary")
    assert attempts == 2
    assert call.retries == 1


@pytest.mark.parametrize("provider", ["openai", "gemini"])
def test_provider_missing_api_key(provider):
    with pytest.raises(ValueError, match="API_KEY"):
        ProviderClient(provider, "", "model")
