"""Merge completed benchmark runs without repeating model or API calls."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jev_rank.reporting import (
    build_report,
    create_chart,
    promote,
    write_json,
    write_jsonl,
    write_metrics_csv,
)


OUTPUT_FILES = (
    "config.json", "metrics.json", "metrics.csv", "pair_scores.jsonl",
    "top_documents.jsonl", "error_analysis.json", "retrieval_ceiling.png",
    "benchmark_report.md",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def unique_rows(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        result[tuple(row[field] for field in fields)] = row
    return list(result.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_run", type=Path)
    parser.add_argument("extension_run", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()

    base, extension = args.base_run.resolve(), args.extension_run.resolve()
    base_config, extension_config = load_json(base / "config.json"), load_json(extension / "config.json")
    for field in ("dataset_repository", "split", "dataset_fingerprint", "query_count", "pair_count", "rerank_k", "at_k"):
        if base_config[field] != extension_config[field]:
            raise ValueError(f"Cannot merge runs: {field} differs")

    base_metrics, extension_metrics = load_json(base / "metrics.json"), load_json(extension / "metrics.json")
    metrics = {**base_metrics["methods"], **extension_metrics["methods"]}
    efficiencies = {**base_metrics["efficiency"], **extension_metrics["efficiency"]}
    if len(metrics) != 9:
        raise ValueError(f"Expected nine unique methods, found {sorted(metrics)}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-merged"
    run_dir = args.output_root.resolve() / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    methods = list(base_config["methods"]) + [m for m in extension_config["methods"] if m not in base_config["methods"]]
    config = {**base_config, **extension_config, "run_id": run_id, "methods": methods, "merged_from": [base.name, extension.name]}
    write_json(run_dir / "config.json", config)

    metrics_payload = {
        "methods": metrics,
        "efficiency": efficiencies,
        "candidate_recall_at_rerank_k": base_metrics["candidate_recall_at_rerank_k"],
        "candidate_oracle": base_metrics["candidate_oracle"],
    }
    write_json(run_dir / "metrics.json", metrics_payload)
    write_metrics_csv(run_dir / "metrics.csv", metrics, efficiencies)

    pair_rows = unique_rows(
        load_jsonl(base / "pair_scores.jsonl") + load_jsonl(extension / "pair_scores.jsonl"),
        ("query_id", "document_id", "method"),
    )
    top_rows = unique_rows(
        load_jsonl(base / "top_documents.jsonl") + load_jsonl(extension / "top_documents.jsonl"),
        ("query_id", "method"),
    )
    write_jsonl(run_dir / "pair_scores.jsonl", pair_rows)
    write_jsonl(run_dir / "top_documents.jsonl", top_rows)

    base_analysis = load_json(base / "error_analysis.json")
    extension_analysis = load_json(extension / "error_analysis.json")
    selection = base_analysis.get("qualitative_analysis", {}).get("selection")
    cases = []
    if selection:
        top_by_query: dict[str, list[dict[str, Any]]] = {}
        for row in top_rows:
            top_by_query.setdefault(row["query_id"], []).append(row)
        for kind, query_id in selection.items():
            exemplar = top_by_query[query_id][0]
            cases.append({
                "kind": kind, "query_id": query_id, "query": exemplar["query"],
                "relevant_ids": exemplar["relevant_ids"], "methods": top_by_query[query_id],
            })
    errors = base_analysis.get("errors", []) + extension_analysis.get("errors", [])
    analysis = {
        "complete": not errors,
        "incomplete_methods": [],
        "errors": errors,
        "qualitative_analysis": {"selection": selection, "cases": cases},
        "merged_from": [base.name, extension.name],
    }
    write_json(run_dir / "error_analysis.json", analysis)

    oracle = metrics_payload["candidate_oracle"]
    chart_error = create_chart(run_dir / "retrieval_ceiling.png", metrics, oracle)
    if chart_error and not (run_dir / "retrieval_ceiling.png").exists():
        (run_dir / "retrieval_ceiling.png").write_bytes(b"")
    report = build_report(
        config, metrics, efficiencies, oracle, metrics_payload["candidate_recall_at_rerank_k"],
        top_rows, selection, chart_error, errors,
    )
    (run_dir / "benchmark_report.md").write_text(report, encoding="utf-8")
    promote(run_dir, args.output_root.resolve() / "results", OUTPUT_FILES)
    print(run_dir)


if __name__ == "__main__":
    main()
