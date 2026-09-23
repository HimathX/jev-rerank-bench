"""Generate publication-ready figures from a completed benchmark run."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = [
    "bm25", "minilm-l4", "minilm-l6", "jev-binary", "jev-graded",
    "openai-binary", "openai-graded", "gemini-binary", "gemini-graded",
]
METHOD_LABELS = {
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
COLORS = {
    "bm25": "#6B7280",
    "minilm-l4": "#D97706",
    "minilm-l6": "#2563EB",
    "jev-binary": "#059669",
    "jev-graded": "#7C3AED",
    "openai-binary": "#0F766E",
    "openai-graded": "#14B8A6",
    "gemini-binary": "#DC2626",
    "gemini-graded": "#F59E0B",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png")
    fig.savefig(output_dir / f"{stem}.svg")
    plt.close(fig)


def quality_figure(metrics: dict[str, Any], output_dir: Path) -> None:
    method_metrics = metrics["methods"]
    methods = [method for method in METHOD_ORDER if method in method_metrics]
    dimensions = [("ndcg", "nDCG@10"), ("mrr", "MRR@10"), ("map", "MAP@20"), ("recall", "Recall@10")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.2), sharex=True)
    y = np.arange(len(methods))
    for axis, (key, title) in zip(axes.flat, dimensions):
        values = [method_metrics[method][key] for method in methods]
        axis.barh(y, values, color=[COLORS[m] for m in methods], height=0.66)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlim(0, 1.0)
        axis.set_yticks(y, [METHOD_LABELS[m] for m in methods])
        axis.invert_yaxis()
        axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
        axis.set_axisbelow(True)
        for row, value in enumerate(values):
            axis.text(value + 0.014, row, f"{value:.3f}", va="center", fontsize=9)
    fig.suptitle("NanoNQ reranking quality", fontsize=16, fontweight="bold", x=0.08, ha="left")
    fig.text(0.08, 0.925, "50 queries · shared BM25 top-20 candidates · higher is better", color="#4B5563")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, output_dir, "quality-metrics")


def efficiency_figure(metrics: dict[str, Any], output_dir: Path) -> None:
    efficiencies = metrics["efficiency"]
    methods = [method for method in METHOD_ORDER if method in efficiencies]
    labels = [METHOD_LABELS[m] for m in methods]
    colors = [COLORS[m] for m in methods]
    total_seconds = [efficiencies[m]["total_seconds"] for m in methods]
    throughput = [efficiencies[m]["pairs_per_second"] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    y = np.arange(len(methods))
    axes[0].barh(y, total_seconds, color=colors, height=0.66)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Total runtime (seconds)")
    axes[0].set_title("End-to-end runtime", loc="left", fontweight="bold")
    axes[0].grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axes[0].set_axisbelow(True)
    for row, value in enumerate(total_seconds):
        axes[0].text(value + max(total_seconds) * 0.018, row, f"{value:.1f}s", va="center", fontsize=9)

    reranker_methods = methods[1:]
    reranker_throughput = throughput[1:]
    reranker_y = np.arange(len(reranker_methods))
    axes[1].barh(reranker_y, reranker_throughput, color=colors[1:], height=0.66)
    axes[1].set_yticks(reranker_y, [METHOD_LABELS[m] for m in reranker_methods])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Scored pairs per second")
    axes[1].set_title("Observed reranker throughput", loc="left", fontweight="bold")
    axes[1].grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axes[1].set_axisbelow(True)
    for row, value in enumerate(reranker_throughput):
        axes[1].text(value + max(reranker_throughput) * 0.025, row, f"{value:.1f}", va="center", fontsize=9)
    fig.suptitle("Benchmark efficiency", fontsize=16, fontweight="bold", x=0.08, ha="left")
    fig.text(0.08, 0.90, "Model loading is included in total runtime; Jev throughput reflects 16 concurrent live calls", color="#4B5563")
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    _save(fig, output_dir, "efficiency-comparison")


def api_operational_figure(metrics: dict[str, Any], output_dir: Path) -> None:
    efficiencies = metrics["efficiency"]
    methods = [method for method in METHOD_ORDER if method.endswith(("-binary", "-graded")) and method in efficiencies]
    labels = [METHOD_LABELS[m] for m in methods]
    x = np.arange(len(methods))
    p50 = [efficiencies[m]["p50_pair_latency_seconds"] for m in methods]
    p95 = [efficiencies[m]["p95_pair_latency_seconds"] for m in methods]
    costs = []
    tokens = []
    for method in methods:
        row = efficiencies[method]
        cached_cost = (
            row.get("cached_input_tokens", 0) / 1_000_000 * row.get("input_price_per_million", 0.0)
            + row.get("cached_output_tokens", 0) / 1_000_000 * row.get("output_price_per_million", 0.0)
        )
        costs.append(row["estimated_cost"] + cached_cost)
        tokens.append(row["input_tokens"] + row.get("cached_input_tokens", 0))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    width = 0.34
    axes[0].bar(x - width / 2, p50, width, label="p50", color="#93C5FD")
    axes[0].bar(x + width / 2, p95, width, label="p95", color="#2563EB")
    axes[0].set_xticks(x, labels)
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_ylabel("Latency per uncached pair (seconds)")
    axes[0].set_title("Live API latency", loc="left", fontweight="bold")
    axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axes[0].set_axisbelow(True)
    axes[0].legend(frameon=False)
    for position, value in zip(x - width / 2, p50):
        axes[0].text(position, value + 0.025, f"{value:.3f}", ha="center", fontsize=9)
    for position, value in zip(x + width / 2, p95):
        axes[0].text(position, value + 0.025, f"{value:.3f}", ha="center", fontsize=9)

    axes[1].bar(x, costs, color=[COLORS[m] for m in methods], width=0.58)
    axes[1].set_xticks(x, labels)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_ylabel("Estimated input + output cost (USD)")
    axes[1].set_title("Cost for 1,000 scored pairs", loc="left", fontweight="bold")
    axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axes[1].set_axisbelow(True)
    axes[1].ticklabel_format(axis="y", style="plain")
    for position, cost, token_count in zip(x, costs, tokens):
        axes[1].text(position, cost + 0.00045, f"${cost:.4f}\n{token_count:,} tokens", ha="center", fontsize=9)
    total_calls = sum(efficiencies[method]["api_calls"] for method in methods)
    total_cached = sum(efficiencies[method].get("cache_hits", 0) for method in methods)
    total_pairs = sum(
        efficiencies[method].get("successful_uncached_pairs", 0)
        + efficiencies[method].get("cache_hits", 0)
        for method in methods
    )
    total_retries = sum(efficiencies[method]["retries"] for method in methods)
    fig.suptitle("API reranker operational profile", fontsize=16, fontweight="bold", x=0.07, ha="left")
    fig.text(
        0.07,
        0.90,
        f"{total_pairs:,} scored pairs | {total_calls:,} calls in completed runs | "
        f"{total_cached:,} resumed from cache | {total_retries:,} retries",
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    _save(fig, output_dir, "api-latency-cost")


def reciprocal_rank_change_figure(pair_scores: list[dict[str, Any]], output_dir: Path) -> None:
    rankings: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in pair_scores:
        rankings[row["query_id"]][row["method"]].append(row)

    def rr(rows: list[dict[str, Any]]) -> float:
        relevant_ranks = [int(row["rank"]) for row in rows if row["relevant"] and int(row["rank"]) <= 10]
        return 1.0 / min(relevant_ranks) if relevant_ranks else 0.0

    observations = []
    for query_id, by_method in rankings.items():
        base = rr(by_method["bm25"])
        observations.append(
            (query_id, rr(by_method["jev-binary"]) - base, rr(by_method["jev-graded"]) - base)
        )
    observations.sort(key=lambda row: (row[2], row[1], row[0]))
    query_ids = [row[0] for row in observations]
    binary = np.array([row[1] for row in observations])
    graded = np.array([row[2] for row in observations])
    positions = np.arange(len(observations))

    fig, axis = plt.subplots(figsize=(12, 5.8))
    axis.axhline(0, color="#374151", linewidth=1)
    axis.vlines(positions, binary, graded, color="#D1D5DB", linewidth=1)
    axis.scatter(positions, binary, s=31, color=COLORS["jev-binary"], marker="o", label="Jev binary", zorder=3)
    axis.scatter(positions, graded, s=35, color=COLORS["jev-graded"], marker="D", label="Jev graded", zorder=3)
    axis.set_ylabel("Change in reciprocal rank@10 vs BM25")
    axis.set_xlabel("Queries sorted by Jev graded change")
    axis.set_title("Per-query Jev movement relative to BM25", loc="left", fontsize=16, fontweight="bold")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, ncol=2, loc="upper left")
    ticks = np.linspace(0, len(query_ids) - 1, 6, dtype=int)
    axis.set_xticks(ticks, [query_ids[index] for index in ticks], rotation=30, ha="right")
    improved = int(np.sum(graded > 0))
    unchanged = int(np.sum(graded == 0))
    regressed = int(np.sum(graded < 0))
    axis.text(
        0.99,
        0.04,
        f"Jev graded: {improved} improved · {unchanged} unchanged · {regressed} regressed",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        color="#4B5563",
    )
    fig.tight_layout()
    _save(fig, output_dir, "per-query-rr-change")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path, nargs="?", default=Path("artifacts/results"))
    args = parser.parse_args()
    output_dir = args.results_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    metrics = _load_json(output_dir / "metrics.json")
    pair_scores = _load_jsonl(output_dir / "pair_scores.jsonl")
    quality_figure(metrics, output_dir)
    efficiency_figure(metrics, output_dir)
    api_operational_figure(metrics, output_dir)
    reciprocal_rank_change_figure(pair_scores, output_dir)
    print(f"Wrote 8 figure files to {output_dir}")


if __name__ == "__main__":
    main()
