from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

from .models import MethodResult, QueryExample

DISPLAY_NAMES = {
    "bm25": "BM25",
    "minilm-l4": "MiniLM L4",
    "minilm-l6": "MiniLM L6",
    "jev-binary": "Jev binary",
    "jev-graded": "Jev graded",
    "openai-binary": "OpenAI binary",
    "openai-graded": "OpenAI graded",
    "gemini-binary": "Gemini binary",
    "gemini-graded": "Gemini graded",
}


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def efficiency(result: MethodResult, input_price: float, output_price: float = 0.0) -> dict[str, Any]:
    live_time = result.live_call_seconds
    input_cost = result.input_tokens / 1_000_000 * input_price
    output_cost = result.output_tokens / 1_000_000 * output_price
    cost = input_cost + output_cost
    total_seen = result.cache_hits + result.successful_uncached_pairs
    return {
        "total_seconds": result.total_seconds,
        "mean_pair_latency_seconds": mean(result.pair_latencies) if result.pair_latencies else 0.0,
        "p50_pair_latency_seconds": median(result.pair_latencies) if result.pair_latencies else 0.0,
        "p95_pair_latency_seconds": percentile(result.pair_latencies, 0.95),
        "pairs_per_second": result.successful_uncached_pairs / live_time if live_time else 0.0,
        "api_calls": result.api_calls,
        "cache_hits": result.cache_hits,
        "cache_hit_rate": result.cache_hits / total_seen if total_seen else 0.0,
        "successful_uncached_pairs": result.successful_uncached_pairs,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cached_input_tokens": result.cached_input_tokens,
        "cached_output_tokens": result.cached_output_tokens,
        "estimated_cost": cost,
        "estimated_input_cost": input_cost,
        "estimated_output_cost": output_cost,
        "input_price_per_million": input_price,
        "output_price_per_million": output_price,
        "estimated_cost_per_paid_pair": cost / result.successful_uncached_pairs if result.successful_uncached_pairs else 0.0,
        "retries": result.retries,
        "resolved_models": sorted(result.resolved_models),
    }


def top_document_records(
    examples: Sequence[QueryExample], results: Mapping[str, MethodResult], top_n: int = 3
) -> list[dict[str, Any]]:
    by_query = {e.query_id: e for e in examples}
    records = []
    for method, result in results.items():
        for query_id, ranking in result.rankings.items():
            example = by_query[query_id]
            text_by_id = {c.document_id: c.text for c in example.candidates}
            documents = [
                {
                    "rank": rank,
                    "document_id": row.document_id,
                    "score": row.score,
                    "original_bm25_position": row.bm25_position,
                    "relevant": row.document_id in example.relevant_ids,
                    "excerpt": " ".join(text_by_id[row.document_id].split())[:240],
                }
                for rank, row in enumerate(ranking[:top_n], 1)
            ]
            records.append({"query_id": query_id, "method": method, "query": example.text, "relevant_ids": sorted(example.relevant_ids), "documents": documents})
    return records


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_metrics_csv(path: Path, metrics: Mapping[str, Mapping[str, float]], efficiencies: Mapping[str, Mapping[str, Any]]) -> None:
    fields = ["method", "ndcg_at_k", "mrr_at_k", "map_at_rerank_k", "recall_at_k", "total_seconds", "mean_pair_latency_seconds", "p50_pair_latency_seconds", "p95_pair_latency_seconds", "pairs_per_second", "api_calls", "estimated_cost", "estimated_cost_per_paid_pair"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, values in metrics.items():
            efficiency_values = efficiencies[method]
            writer.writerow({
                "method": method,
                "ndcg_at_k": values["ndcg"], "mrr_at_k": values["mrr"], "map_at_rerank_k": values["map"], "recall_at_k": values["recall"],
                **{key: efficiency_values[key] for key in fields[5:]},
            })


def create_chart(path: Path, metrics: Mapping[str, Mapping[str, float]], oracle: Mapping[str, float]) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [DISPLAY_NAMES.get(name, name) for name in metrics] + ["Qrel candidate oracle"]
        values = [row["ndcg"] for row in metrics.values()] + [oracle["ndcg"]]
        fig, axis = plt.subplots(figsize=(9, 4.8))
        colors = ["#4472C4"] * len(metrics) + ["#70AD47"]
        axis.bar(labels, values, color=colors)
        axis.set_ylabel("nDCG at evaluation cutoff")
        axis.set_ylim(0, 1)
        axis.set_title("Reranking quality within the shared BM25 candidate ceiling")
        axis.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return None
    except Exception as exc:  # chart failure must not fail the benchmark
        return f"{type(exc).__name__}: {exc}"


def _metric_table(metrics: Mapping[str, Mapping[str, float]], at_k: int, rerank_k: int) -> str:
    lines = [f"| Method | nDCG@{at_k} | MRR@{at_k} | MAP@{rerank_k} | Recall@{at_k} |", "|---|---:|---:|---:|---:|"]
    for method, row in metrics.items():
        lines.append(f"| {DISPLAY_NAMES.get(method, method)} | {row['ndcg']:.4f} | {row['mrr']:.4f} | {row['map']:.4f} | {row['recall']:.4f} |")
    return "\n".join(lines)


def build_report(
    config: Mapping[str, Any], metrics: Mapping[str, Mapping[str, float]], efficiencies: Mapping[str, Mapping[str, Any]],
    oracle: Mapping[str, float], ceiling: float, top_records: Sequence[Mapping[str, Any]], cases: Mapping[str, str] | None,
    chart_error: str | None, errors: Sequence[Mapping[str, Any]],
) -> str:
    at_k, rerank_k = config["at_k"], config["rerank_k"]
    lines = [
        "# NanoBEIR reranking benchmark", "",
        f"Run date: `{config['run_date']}`  ", f"Status: **{'complete' if not errors else 'incomplete'}**  ",
        f"Dataset: `{config['dataset_repository']}` / `{config['split']}`  ", f"Dataset fingerprint: `{config['dataset_fingerprint']}`  ",
        f"Queries: {config['query_count']}; query-document pairs: {config['pair_count']}; BM25 candidate depth: {rerank_k}.", "",
        "## Retrieval ceiling", "", f"Shared BM25 candidate Recall@{rerank_k}: **{ceiling:.4f}**. All methods receive exactly these candidates; missing positives are not injected.", "",
        "Qrel-based candidate-set oracle (evaluation-only, unattainable under available judgments; not a model): ", "",
        f"nDCG@{at_k} {oracle['ndcg']:.4f}; MRR@{at_k} {oracle['mrr']:.4f}; MAP@{rerank_k} {oracle['map']:.4f}; Recall@{at_k} {oracle['recall']:.4f}.", "",
        "## Main metrics", "", _metric_table(metrics, at_k, rerank_k), "",
        "## Latency and cost", "", "API latency percentiles and live throughput use uncached requests only. Estimated cost is `input_tokens / 1,000,000 * input price + output_tokens / 1,000,000 * output price`. Jev output price is zero; OpenAI and Gemini output tokens are included.", "",
        "| Method | Total s | Mean s | p50 s | p95 s | Pairs/s | API calls | New input/output tokens | Est. cost | Cost/paid pair | Cache hits (rate) | Retries |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in efficiencies.items():
        lines.append(f"| {DISPLAY_NAMES.get(method, method)} | {row['total_seconds']:.3f} | {row['mean_pair_latency_seconds']:.3f} | {row['p50_pair_latency_seconds']:.3f} | {row['p95_pair_latency_seconds']:.3f} | {row['pairs_per_second']:.2f} | {row['api_calls']} | {row['input_tokens']}/{row['output_tokens']} | ${row['estimated_cost']:.6f} | ${row['estimated_cost_per_paid_pair']:.6f} | {row['cache_hits']} ({row['cache_hit_rate']:.1%}) | {row['retries']} |")
    resolved_versions = sorted({version for row in efficiencies.values() for version in row["resolved_models"]})
    cached_tokens = sum(row["cached_input_tokens"] for row in efficiencies.values()), sum(row["cached_output_tokens"] for row in efficiencies.values())
    lines += ["", f"Cost-per-pair denominator is successful uncached pairs. Configured concurrency: {config['concurrency']}. Cached token metadata (excluded from new cost): {cached_tokens[0]} input / {cached_tokens[1]} output.", "", f"Requested models: Jev `{config['jev_model']}`, OpenAI `{config['openai_model']}`, Gemini `{config['gemini_model']}`. Resolved versions present: {', '.join(f'`{v}`' for v in resolved_versions) if resolved_versions else 'none (no API calls)'}." ]
    lines += ["", f"Configured per-million-token prices — Jev input/output: ${config['jev_input_price_per_million']}/$0; OpenAI: ${config['openai_input_price_per_million']}/${config['openai_output_price_per_million']}; Gemini: ${config['gemini_input_price_per_million']}/${config['gemini_output_price_per_million']}."]
    if chart_error:
        lines += ["", f"Plotting failed, but numeric results were preserved: `{chart_error}`"]
    lines += ["", "## Top-three comparison", "", "Scores are method-specific and should only be compared within a method.", ""]
    for record in top_records:
        lines += [f"### {record['query_id']} — {DISPLAY_NAMES.get(record['method'], record['method'])}", "", f"Query: {record['query']}", "", "| Rank | Document | Score | BM25 pos. | Relevant | Excerpt |", "|---:|---|---:|---:|---|---|"]
        for doc in record["documents"]:
            excerpt = str(doc["excerpt"]).replace("|", "\\|")
            lines.append(f"| {doc['rank']} | `{doc['document_id']}` | {doc['score']:.6f} | {doc['original_bm25_position']} | {doc['relevant']} | {excerpt} |")
        lines.append("")
    lines += ["## Deterministic qualitative diagnostics", "", "These predeclared, qrel-based selected cases are diagnostic—not representative statistical evidence. OpenAI and Gemini probabilities/scores are model-reported ranking values, not calibrated probabilities.", ""]
    if cases:
        labels = {"largest_jev_graded_improvement": "Largest Jev graded improvement", "largest_jev_graded_regression": "Largest Jev graded regression", "strongest_jev_disagreement": "Strongest Jev binary/graded Kendall disagreement"}
        record_map = {(r["query_id"], r["method"]): r for r in top_records}
        for key, query_id in cases.items():
            exemplar = next(r for r in top_records if r["query_id"] == query_id)
            relevant = next((r.get("relevant_ids", []) for r in top_records if r["query_id"] == query_id), [])
            lines += [f"### {labels[key]}: `{query_id}`", "", f"Query: {exemplar['query']}", "", f"Relevant qrel IDs: {', '.join(map(str, relevant)) if relevant else '(see machine-readable analysis)'}.", ""]
            for method in metrics:
                rec = record_map[(query_id, method)]
                compact = "; ".join(f"#{d['rank']} {d['document_id']} ({d['score']:.4f}, BM25 {d['original_bm25_position']}, rel={d['relevant']}) — {d['excerpt']}" for d in rec["documents"])
                lines += [f"- **{DISPLAY_NAMES.get(method, method)}:** {compact}", ""]
    else:
        lines += ["Not available unless BM25 and both Jev formulations are included.", ""]
    lines += ["## Errors and retries", "", f"Errors: {len(errors)}. See `error_analysis.json` for details. Retry counts are shown in the efficiency table.", "", "## Limitations", "", "NanoNQ is a 50-query pilot with potentially incomplete binary relevance judgments. Results do not establish statistical significance or probability calibration. The qrel oracle is an evaluation diagnostic, not a runnable ranker. Selected cases must not be used to claim general superiority. Throughput varies with rate limits, server load, network conditions, and local scheduling; concurrency does not imply linear scaling.", ""]
    return "\n".join(lines)


def promote(run_dir: Path, results_dir: Path, filenames: Sequence[str]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        shutil.copy2(run_dir / filename, results_dir / filename)
