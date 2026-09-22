from __future__ import annotations

import json

import httpx
import pytest

from jev_rank.cache import JsonlCache, cache_key
from jev_rank.jev_client import JevClient, JevResponseError, build_payload, parse_response


def test_binary_and_graded_parsing():
    binary = parse_response({"model": "jev-1.2", "answers": {"relevant": {"type": "noul", "noul": 0.8}}, "usage": {"input_tokens": 12, "output_tokens": 2}}, "jev-binary")
    assert binary.score == 0.8
    graded = parse_response({"model": "jev-1.2", "answers": {"relevance": {"type": "score", "score": 3.5, "probabilities": {"3": 0.5}, "confidence": 0.7}}, "usage": {"input_tokens": 20, "output_tokens": 3}}, "jev-graded")
    assert graded.score == 3.5
    assert graded.probabilities == {"3": 0.5}
    assert graded.confidence == 0.7


@pytest.mark.parametrize("body", [{}, {"model": "v", "answers": {}, "usage": {}}, {"model": "v", "answers": {"relevant": {"noul": "yes"}}, "usage": {}}])
def test_malformed_responses(body):
    with pytest.raises(JevResponseError):
        parse_response(body, "jev-binary")


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 529])
async def test_retry_handling(status):
    attempts = 0
    sleeps = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(status, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"model": "jev-1", "answers": {"relevant": {"type": "noul", "noul": 0.9}}, "usage": {"input_tokens": 4, "output_tokens": 1}})

    async def no_sleep(delay: float):
        sleeps.append(delay)

    async with JevClient("secret", max_retries=2, transport=httpx.MockTransport(handler), sleep=no_sleep) as client:
        call = await client.score(build_payload("jev-binary", "q", "d", "jev-latest"), "jev-binary")
    assert call.retries == 1
    assert attempts == 2
    assert sleeps == [0.0]


def test_missing_api_key():
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        JevClient("")


@pytest.mark.asyncio
async def test_cache_hit_resume_and_truncated_tail(tmp_path):
    path = tmp_path / "cache.jsonl"
    cache = JsonlCache(path)
    key = cache_key("repo", "fp", "split", "q", "d", "jev-binary", {"x": 1}, "jev-latest")
    await cache.append({"cache_key": key, "success": True, "ranking_score": 0.7})
    await cache.append({"cache_key": "failed", "success": False, "error": "x"})
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"partial":')
    reloaded = JsonlCache(path)
    reloaded.load()
    assert reloaded.get(key)["ranking_score"] == 0.7
    assert reloaded.get("failed") is None

