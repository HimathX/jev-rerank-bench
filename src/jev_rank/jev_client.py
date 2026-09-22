from __future__ import annotations

import asyncio
import email.utils
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

import httpx

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
RETRYABLE_STATUS = {429, 500, 502, 503, 504, 529}
BINARY_INSTRUCTIONS = "Does the document contain information that directly helps answer the query?"
BINARY_CRITERIA = {
    "true": "The document provides evidence needed to answer the query.",
    "false": "The document does not provide evidence needed to answer the query.",
}
GRADED_INSTRUCTIONS = "Rate how useful the document is for correctly answering the query."
GRADED_CRITERIA = [
    "Irrelevant",
    "Related topic but does not help answer the query",
    "Partially useful",
    "Directly useful",
    "Contains essential answer evidence",
]


class JevError(RuntimeError):
    def __init__(self, message: str, attempts: int = 0, retries: int = 0):
        super().__init__(message)
        self.attempts = attempts
        self.retries = retries


class JevResponseError(JevError):
    pass


@dataclass(frozen=True)
class ParsedJevResponse:
    score: float
    answer: dict[str, Any]
    probabilities: Any
    confidence: float | None
    input_tokens: int
    output_tokens: int
    resolved_model: str


@dataclass(frozen=True)
class JevCall:
    parsed: ParsedJevResponse
    latency: float
    retries: int


def build_payload(method: str, query: str, document: str, model: str) -> dict[str, Any]:
    if method == "jev-binary":
        questions = {"relevant": {"type": "noul", "instructions": BINARY_INSTRUCTIONS, "criteria": BINARY_CRITERIA}}
    elif method == "jev-graded":
        questions = {"relevance": {"type": "score", "instructions": GRADED_INSTRUCTIONS, "criteria": GRADED_CRITERIA}}
    else:
        raise ValueError(f"Unknown Jev method: {method}")
    return {"state": {"query": query, "document": document}, "model": model, "questions": questions}


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JevResponseError(f"Jev response field {label} must be numeric")
    return float(value)


def parse_response(data: Mapping[str, Any], method: str) -> ParsedJevResponse:
    if method not in {"jev-binary", "jev-graded"}:
        raise ValueError(f"Unknown Jev method: {method}")
    answers = data.get("answers")
    if not isinstance(answers, dict):
        raise JevResponseError("Jev response is missing object field 'answers'")
    key, primitive = ("relevant", "noul") if method == "jev-binary" else ("relevance", "score")
    answer = answers.get(key)
    if not isinstance(answer, dict) or primitive not in answer:
        raise JevResponseError(f"Jev response is missing answers.{key}.{primitive}")
    usage = data.get("usage", {})
    if not isinstance(usage, dict):
        raise JevResponseError("Jev response field 'usage' must be an object")
    resolved = data.get("model") or data.get("model_version") or data.get("resolved_model")
    if not isinstance(resolved, str) or not resolved:
        raise JevResponseError("Jev response is missing a resolved model version")
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        raise JevResponseError("Jev token usage must contain integer token counts")
    probabilities = answer.get("probabilities", answer.get("distribution"))
    confidence = answer.get("confidence")
    return ParsedJevResponse(
        score=_number(answer[primitive], f"answers.{key}.{primitive}"),
        answer=dict(answer),
        probabilities=probabilities,
        confidence=_number(confidence, f"answers.{key}.confidence") if confidence is not None else None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        resolved_model=resolved,
    )


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            parsed = email.utils.parsedate_to_datetime(value)
            return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError):
            return None


class JevClient:
    def __init__(
        self,
        api_key: str,
        timeout: float = 60.0,
        max_retries: int = 5,
        concurrency: int = 16,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Any] = asyncio.sleep,
    ):
        if not api_key:
            raise ValueError("TYPESAFE_API_KEY is required when a Jev method is selected")
        self.max_retries = max_retries
        self._semaphore = asyncio.Semaphore(concurrency)
        self._sleep = sleep
        self._client = httpx.AsyncClient(
            timeout=timeout,
            transport=transport,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )

    async def __aenter__(self) -> "JevClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self._client.aclose()

    async def score(self, payload: Mapping[str, Any], method: str) -> JevCall:
        async with self._semaphore:
            started = time.perf_counter()
            for attempt in range(self.max_retries + 1):
                try:
                    response = await self._client.post(JEV_ENDPOINT, content=json.dumps(payload, ensure_ascii=False))
                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    if attempt >= self.max_retries:
                        raise JevError(f"Jev request failed after {attempt + 1} attempts: {exc}", attempt + 1, attempt) from exc
                    await self._sleep((2**attempt) + random.uniform(0, 0.25))
                    continue
                if response.status_code in RETRYABLE_STATUS:
                    if attempt >= self.max_retries:
                        raise JevError(f"Jev returned HTTP {response.status_code} after {attempt + 1} attempts", attempt + 1, attempt)
                    delay = _retry_after(response)
                    await self._sleep(delay if delay is not None else (2**attempt) + random.uniform(0, 0.25))
                    continue
                if response.status_code in (401, 403):
                    raise JevError(f"Jev authentication failed (HTTP {response.status_code}); check TYPESAFE_API_KEY", attempt + 1, attempt)
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise JevError(f"Jev request failed with HTTP {response.status_code}: {response.text[:300]}", attempt + 1, attempt) from exc
                try:
                    data = response.json()
                except ValueError as exc:
                    raise JevResponseError("Jev returned invalid JSON") from exc
                return JevCall(parse_response(data, method), time.perf_counter() - started, attempt)
            raise AssertionError("unreachable")
