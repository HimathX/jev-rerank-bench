from __future__ import annotations

from jev_rank.reporting import build_report


def test_report_generation_contains_required_sections():
    config = {"run_date": "2026-01-01", "dataset_repository": "repo", "split": "NanoNQ", "dataset_fingerprint": "fp", "query_count": 1, "pair_count": 2, "at_k": 1, "rerank_k": 2, "jev_input_price_per_million": 0.042, "concurrency": 16, "jev_model": "jev-latest"}
    metrics = {"bm25": {"ndcg": 1.0, "mrr": 1.0, "map": 0.5, "recall": 0.5}}
    efficiency = {"bm25": {"total_seconds": 0.1, "mean_pair_latency_seconds": 0, "p50_pair_latency_seconds": 0, "p95_pair_latency_seconds": 0, "pairs_per_second": 0, "api_calls": 0, "input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0, "cached_output_tokens": 0, "estimated_cost": 0, "estimated_cost_per_paid_pair": 0, "cache_hits": 0, "cache_hit_rate": 0, "retries": 0, "resolved_models": []}}
    report = build_report(config, metrics, efficiency, metrics["bm25"], 0.5, [], None, None, [])
    assert "Retrieval ceiling" in report
    assert "Qrel-based candidate-set oracle" in report
    assert "Limitations" in report
    assert "Cost-per-pair denominator" in report
