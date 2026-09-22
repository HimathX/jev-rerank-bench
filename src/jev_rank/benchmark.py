from __future__ import annotations

import asyncio
import json
import os
import platform
import random
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .cache import JsonlCache, cache_key
from .cross_encoder import DEFAULT_MODELS, run_cross_encoder
from .data import BenchmarkData, load_nanobeir
from .evaluation import bm25_rankings, candidate_recall, evaluate, oracle_rankings, rank_scores, select_qualitative_cases
from .jev_client import BINARY_CRITERIA, BINARY_INSTRUCTIONS, GRADED_CRITERIA, GRADED_INSTRUCTIONS, JevClient, build_payload
from .models import MethodResult, QueryExample
from .reporting import build_report, create_chart, efficiency, promote, top_document_records, write_json, write_jsonl, write_metrics_csv

ALL_METHODS = ("bm25", "minilm-l4", "minilm-l6", "jev-binary", "jev-graded")
OUTPUT_FILES = ("config.json", "metrics.json", "metrics.csv", "pair_scores.jsonl", "top_documents.jsonl", "error_analysis.json", "retrieval_ceiling.png", "benchmark_report.md")


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset: str = "nq"
    rerank_k: int = 20
    at_k: int = 10
    methods: tuple[str, ...] = ALL_METHODS
    limit_queries: int | None = None
    cache_dir: Path = Path("artifacts/cache")
    output_dir: Path = Path("artifacts")
    request_timeout: float = 60.0
    max_retries: int = 5
    concurrency: int = 16
    jev_model: str = "jev-latest"
    jev_input_price_per_million: float = 0.042
    seed: int = 42
    refresh_jev: bool = False
    cross_encoder_batch_size: int = 32
    device: str | None = None
    cross_encoder_models: dict[str, str] | None = None


def _cached_parsed(record: dict[str, Any]) -> tuple[float, int, int, str]:
    return float(record["ranking_score"]), int(record.get("input_tokens", 0)), int(record.get("output_tokens", 0)), str(record["resolved_model_version"])


async def _run_jev_method(
    method: str, data: BenchmarkData, config: BenchmarkConfig, cache: JsonlCache, api_key: str
) -> MethodResult:
    result = MethodResult(method, {})
    scores: dict[str, dict[str, float]] = {example.query_id: {} for example in data.examples}
    started = time.perf_counter()
    live_intervals: list[tuple[float, float]] = []

    async with JevClient(api_key, config.request_timeout, config.max_retries, config.concurrency) as client:
        async def score_one(example: QueryExample, candidate: Any) -> None:
            payload = build_payload(method, example.text, candidate.text, config.jev_model)
            key = cache_key(data.repository, data.fingerprint, data.split, example.query_id, candidate.document_id, method, payload, config.jev_model)
            cached = None if config.refresh_jev else cache.get(key)
            if cached is not None:
                score, input_tokens, output_tokens, version = _cached_parsed(cached)
                scores[example.query_id][candidate.document_id] = score
                result.cache_hits += 1
                result.cached_input_tokens += input_tokens
                result.cached_output_tokens += output_tokens
                result.resolved_models.add(version)
                return
            timestamp = datetime.now(timezone.utc).isoformat()
            base = {
                "cache_key": key, "dataset": data.repository, "dataset_fingerprint": data.fingerprint, "split": data.split,
                "query_id": example.query_id, "document_id": candidate.document_id, "method": method,
                "query_text": example.text, "document_text": candidate.text, "request_payload": payload,
                "requested_model": config.jev_model, "timestamp": timestamp,
            }
            try:
                live_started = time.perf_counter()
                call = await client.score(payload, method)
                live_intervals.append((live_started, time.perf_counter()))
                parsed = call.parsed
                record = {
                    **base, "success": True, "ranking_score": parsed.score, "complete_answer": parsed.answer,
                    "probabilities": parsed.probabilities, "confidence": parsed.confidence,
                    "resolved_model_version": parsed.resolved_model, "input_tokens": parsed.input_tokens,
                    "output_tokens": parsed.output_tokens, "latency_seconds": call.latency, "retries": call.retries,
                }
                await cache.append(record)
                scores[example.query_id][candidate.document_id] = parsed.score
                result.pair_latencies.append(call.latency)
                result.api_calls += 1 + call.retries
                result.successful_uncached_pairs += 1
                result.input_tokens += parsed.input_tokens
                result.output_tokens += parsed.output_tokens
                result.retries += call.retries
                result.resolved_models.add(parsed.resolved_model)
            except Exception as exc:
                live_intervals.append((live_started, time.perf_counter()))
                result.api_calls += int(getattr(exc, "attempts", 1))
                result.retries += int(getattr(exc, "retries", 0))
                error = {**base, "success": False, "error": {"type": type(exc).__name__, "message": str(exc)}}
                await cache.append(error)
                result.errors.append(error)

        tasks = [score_one(example, candidate) for example in data.examples for candidate in example.candidates]
        await asyncio.gather(*tasks)

    for example in data.examples:
        if len(scores[example.query_id]) == len(example.candidates):
            ordered_scores = [scores[example.query_id][c.document_id] for c in example.candidates]
            result.rankings[example.query_id] = rank_scores(ordered_scores, [c.document_id for c in example.candidates])
    result.total_seconds = time.perf_counter() - started
    if live_intervals:
        result.live_call_seconds = max(end for _, end in live_intervals) - min(begin for begin, _ in live_intervals)
    return result


def _pair_score_rows(examples: Sequence[QueryExample], results: dict[str, MethodResult]) -> list[dict[str, Any]]:
    rows = []
    for method, result in results.items():
        for example in examples:
            for rank, document in enumerate(result.rankings.get(example.query_id, []), 1):
                rows.append({
                    "query_id": example.query_id, "document_id": document.document_id, "method": method,
                    "score": document.score, "rank": rank, "original_bm25_position": document.bm25_position,
                    "relevant": document.document_id in example.relevant_ids,
                })
    return rows


def run_benchmark(config: BenchmarkConfig) -> Path:
    random.seed(config.seed)
    invalid = sorted(set(config.methods) - set(ALL_METHODS))
    if invalid:
        raise ValueError(f"Unknown methods: {', '.join(invalid)}. Valid methods: {', '.join(ALL_METHODS)}")
    if config.rerank_k <= 0 or config.at_k <= 0:
        raise ValueError("--rerank-k and --at-k must be positive")
    jev_methods = [method for method in config.methods if method.startswith("jev-")]
    api_key = os.environ.get("TYPESAFE_API_KEY", "")
    if jev_methods and not api_key:
        raise ValueError("TYPESAFE_API_KEY is required when Jev methods are selected; BM25 and MiniLM methods work without it")

    data = load_nanobeir(config.dataset, config.rerank_k, config.limit_queries)
    if not data.examples:
        raise ValueError("Dataset produced no queries")
    pair_count = sum(len(example.candidates) for example in data.examples)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = config.output_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    cache = JsonlCache(config.cache_dir / "jev-results.jsonl")
    cache.load()
    model_ids = dict(DEFAULT_MODELS)
    model_ids.update(config.cross_encoder_models or {})
    config_record = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "methods": list(config.methods), "run_id": run_id, "run_date": datetime.now(timezone.utc).date().isoformat(),
        "dataset_repository": data.repository, "split": data.split, "dataset_fingerprint": data.fingerprint,
        "query_count": len(data.examples), "pair_count": pair_count, "cross_encoder_model_ids": model_ids,
        "python_version": sys.version, "platform": platform.platform(),
        "jev_prompts": {
            "binary": {"instructions": BINARY_INSTRUCTIONS, "criteria": BINARY_CRITERIA},
            "graded": {"instructions": GRADED_INSTRUCTIONS, "criteria": GRADED_CRITERIA},
        },
    }
    write_json(run_dir / "config.json", config_record)

    results: dict[str, MethodResult] = {}
    if "bm25" in config.methods:
        started = time.perf_counter()
        rankings = bm25_rankings(data.examples)
        total = time.perf_counter() - started
        per_pair = total / pair_count if pair_count else 0.0
        results["bm25"] = MethodResult("bm25", rankings, [per_pair] * pair_count, total, successful_uncached_pairs=pair_count, live_call_seconds=total)
    for method in ("minilm-l4", "minilm-l6"):
        if method in config.methods:
            results[method] = run_cross_encoder(method, model_ids[method], data.examples, config.cross_encoder_batch_size, config.device)
    for method in ("jev-binary", "jev-graded"):
        if method in config.methods:
            results[method] = asyncio.run(_run_jev_method(method, data, config, cache, api_key))

    all_errors = [error for result in results.values() for error in result.errors]
    incomplete_methods = [method for method, result in results.items() if len(result.rankings) != len(data.examples)]
    errors_payload = {"complete": not all_errors and not incomplete_methods, "incomplete_methods": incomplete_methods, "errors": all_errors}
    write_json(run_dir / "error_analysis.json", errors_payload)

    complete_results = {method: result for method, result in results.items() if len(result.rankings) == len(data.examples)}
    metrics = {method: evaluate(result.rankings, data.examples, config.at_k, config.rerank_k) for method, result in complete_results.items()}
    efficiencies = {method: efficiency(result, config.jev_input_price_per_million) for method, result in complete_results.items()}
    oracle = evaluate(oracle_rankings(data.examples), data.examples, config.at_k, config.rerank_k)
    ceiling = candidate_recall(data.examples)
    metrics_payload = {"methods": metrics, "efficiency": efficiencies, "candidate_recall_at_rerank_k": ceiling, "candidate_oracle": oracle}
    write_json(run_dir / "metrics.json", metrics_payload)
    write_metrics_csv(run_dir / "metrics.csv", metrics, efficiencies)
    write_jsonl(run_dir / "pair_scores.jsonl", _pair_score_rows(data.examples, complete_results))
    top_records = top_document_records(data.examples, complete_results)
    write_jsonl(run_dir / "top_documents.jsonl", top_records)
    cases = select_qualitative_cases(data.examples, {name: result.rankings for name, result in complete_results.items()}, config.at_k)
    case_payload = {"selection": cases, "cases": []}
    if cases:
        examples_by_id = {example.query_id: example for example in data.examples}
        records_by_query = {}
        for query_id in set(cases.values()):
            records_by_query[query_id] = {
                "query": examples_by_id[query_id].text, "relevant_ids": sorted(examples_by_id[query_id].relevant_ids),
                "methods": [record for record in top_records if record["query_id"] == query_id],
            }
        case_payload["cases"] = [{"kind": key, "query_id": qid, **records_by_query[qid]} for key, qid in cases.items()]
    write_json(run_dir / "error_analysis.json", {**errors_payload, "qualitative_analysis": case_payload})
    chart_error = create_chart(run_dir / "retrieval_ceiling.png", metrics, oracle)
    # Always provide the named chart artifact, even when plotting is unavailable.
    if chart_error and not (run_dir / "retrieval_ceiling.png").exists():
        (run_dir / "retrieval_ceiling.png").write_bytes(b"")
    report = build_report(config_record, metrics, efficiencies, oracle, ceiling, top_records, cases, chart_error, all_errors)
    (run_dir / "benchmark_report.md").write_text(report, encoding="utf-8")

    if not all_errors and not incomplete_methods:
        promote(run_dir, config.output_dir / "results", OUTPUT_FILES)
    return run_dir
