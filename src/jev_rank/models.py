from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Candidate:
    document_id: str
    text: str
    bm25_position: int


@dataclass(frozen=True)
class QueryExample:
    query_id: str
    text: str
    candidates: tuple[Candidate, ...]
    relevant_ids: frozenset[str]


@dataclass
class RankedDocument:
    document_id: str
    score: float
    bm25_position: int


@dataclass
class MethodResult:
    name: str
    rankings: dict[str, list[RankedDocument]]
    pair_latencies: list[float] = field(default_factory=list)
    total_seconds: float = 0.0
    api_calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cached_output_tokens: int = 0
    retries: int = 0
    resolved_models: set[str] = field(default_factory=set)
    errors: list[dict[str, Any]] = field(default_factory=list)
    successful_uncached_pairs: int = 0
    live_call_seconds: float = 0.0
