from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .models import QueryExample, RankedDocument


def rank_scores(scores: Sequence[float], document_ids: Sequence[str]) -> list[RankedDocument]:
    if len(scores) != len(document_ids):
        raise ValueError("scores and document_ids must have the same length")
    rows = [RankedDocument(doc_id, float(score), position) for position, (doc_id, score) in enumerate(zip(document_ids, scores), 1)]
    return sorted(rows, key=lambda row: (-row.score, row.bm25_position))


def reciprocal_rank(ranking: Sequence[RankedDocument], relevant: set[str] | frozenset[str], at_k: int) -> float:
    return next((1.0 / rank for rank, row in enumerate(ranking[:at_k], 1) if row.document_id in relevant), 0.0)


def query_metrics(ranking: Sequence[RankedDocument], relevant: set[str] | frozenset[str], at_k: int, map_k: int) -> dict[str, float]:
    gains = [1 if row.document_id in relevant else 0 for row in ranking[:at_k]]
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))
    ideal_count = min(len(relevant), at_k)
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    hits = 0
    precision_sum = 0.0
    for rank, row in enumerate(ranking[:map_k], 1):
        if row.document_id in relevant:
            hits += 1
            precision_sum += hits / rank
    denominator = len(relevant)
    return {
        "ndcg": dcg / idcg if idcg else 0.0,
        "mrr": reciprocal_rank(ranking, relevant, at_k),
        "map": precision_sum / denominator if denominator else 0.0,
        "recall": sum(gains) / denominator if denominator else 0.0,
    }


def evaluate(rankings: Mapping[str, Sequence[RankedDocument]], examples: Sequence[QueryExample], at_k: int, map_k: int) -> dict[str, float]:
    values = [query_metrics(rankings[e.query_id], e.relevant_ids, at_k, map_k) for e in examples]
    if not values:
        raise ValueError("Cannot evaluate zero queries")
    return {name: sum(row[name] for row in values) / len(values) for name in ("ndcg", "mrr", "map", "recall")}


def bm25_rankings(examples: Sequence[QueryExample]) -> dict[str, list[RankedDocument]]:
    return {
        example.query_id: [RankedDocument(c.document_id, float(-c.bm25_position), c.bm25_position) for c in example.candidates]
        for example in examples
    }


def oracle_rankings(examples: Sequence[QueryExample]) -> dict[str, list[RankedDocument]]:
    result = {}
    for example in examples:
        ordered = sorted(example.candidates, key=lambda c: (c.document_id not in example.relevant_ids, c.bm25_position))
        result[example.query_id] = [RankedDocument(c.document_id, float(c.document_id in example.relevant_ids), c.bm25_position) for c in ordered]
    return result


def candidate_recall(examples: Sequence[QueryExample]) -> float:
    values = [len({c.document_id for c in e.candidates} & e.relevant_ids) / len(e.relevant_ids) if e.relevant_ids else 0.0 for e in examples]
    return sum(values) / len(values) if values else 0.0


def kendall_distance(a: Sequence[RankedDocument], b: Sequence[RankedDocument]) -> float:
    if len(a) < 2:
        return 0.0
    positions = {row.document_id: i for i, row in enumerate(b)}
    order = [positions[row.document_id] for row in a]
    inversions = sum(order[i] > order[j] for i in range(len(order)) for j in range(i + 1, len(order)))
    return inversions / (len(order) * (len(order) - 1) / 2)


def select_qualitative_cases(
    examples: Sequence[QueryExample], rankings: Mapping[str, Mapping[str, Sequence[RankedDocument]]], at_k: int
) -> dict[str, str] | None:
    required = {"bm25", "jev-graded", "jev-binary"}
    if not required.issubset(rankings):
        return None
    changes = []
    for example in examples:
        qid = example.query_id
        bm = query_metrics(rankings["bm25"][qid], example.relevant_ids, at_k, len(example.candidates))
        graded = query_metrics(rankings["jev-graded"][qid], example.relevant_ids, at_k, len(example.candidates))
        changes.append((graded["mrr"] - bm["mrr"], graded["ndcg"] - bm["ndcg"], qid))
    improvement = max(changes, key=lambda x: (x[0], x[1], x[2]))[2]
    regression = min(changes, key=lambda x: (x[0], x[1], x[2]))[2]
    disagreements = [(kendall_distance(rankings["jev-binary"][e.query_id], rankings["jev-graded"][e.query_id]), e.query_id) for e in examples]
    disagreement = max(disagreements, key=lambda x: (x[0], x[1]))[1]
    return {"largest_jev_graded_improvement": improvement, "largest_jev_graded_regression": regression, "strongest_jev_disagreement": disagreement}
