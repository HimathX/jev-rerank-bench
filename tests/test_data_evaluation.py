from __future__ import annotations

import pytest

from jev_rank.data import resolve_split, transform_rows
from jev_rank.evaluation import candidate_recall, evaluate, oracle_rankings, rank_scores, select_qualitative_cases
from jev_rank.models import RankedDocument
from jev_rank.reporting import top_document_records


def sample_examples():
    return transform_rows(
        [{"_id": "d1", "text": "one"}, {"_id": "d2", "text": "two"}, {"_id": "d3", "text": "three"}],
        [{"_id": "q1", "text": "question"}],
        [{"query-id": "q1", "corpus-id": "d2"}, {"query-id": "q1", "corpus-id": "missing"}],
        [{"query-id": "q1", "corpus-ids": ["d1", "d2", "d3"]}],
        rerank_k=3,
    )


def test_schema_transformation_and_unknown_alias():
    examples = sample_examples()
    assert examples[0].candidates[1].document_id == "d2"
    assert examples[0].candidates[1].bm25_position == 2
    assert examples[0].relevant_ids == {"d2", "missing"}
    assert resolve_split("nq") == "NanoNQ"
    with pytest.raises(ValueError, match="case-sensitive"):
        resolve_split("NQ")


def test_schema_missing_required_field_is_clear():
    with pytest.raises(ValueError, match="corpus text"):
        transform_rows([{"_id": "d"}], [], [], [], 1)


def test_tie_breaking_metrics_missing_positive_and_oracle():
    examples = sample_examples()
    ranking = rank_scores([0.5, 0.9, 0.5], ["d1", "d2", "d3"])
    assert [r.document_id for r in ranking] == ["d2", "d1", "d3"]
    metrics = evaluate({"q1": ranking}, examples, at_k=2, map_k=3)
    assert metrics["mrr"] == 1.0
    assert metrics["ndcg"] == pytest.approx(1.0 / (1 + 1 / 1.5849625007))
    assert metrics["map"] == 0.5  # all known qrels form the AP denominator
    assert metrics["recall"] == 0.5
    assert candidate_recall(examples) == 0.5
    oracle = oracle_rankings(examples)["q1"]
    assert [r.document_id for r in oracle] == ["d2", "d1", "d3"]


def test_deterministic_cases_and_top_three():
    examples = sample_examples()
    bm25 = [RankedDocument("d1", -1, 1), RankedDocument("d2", -2, 2), RankedDocument("d3", -3, 3)]
    graded = [RankedDocument("d2", 1, 2), RankedDocument("d1", 0, 1), RankedDocument("d3", 0, 3)]
    binary = [RankedDocument("d3", 1, 3), RankedDocument("d1", 0, 1), RankedDocument("d2", 0, 2)]
    rankings = {"bm25": {"q1": bm25}, "jev-graded": {"q1": graded}, "jev-binary": {"q1": binary}}
    assert select_qualitative_cases(examples, rankings, 2) == {
        "largest_jev_graded_improvement": "q1", "largest_jev_graded_regression": "q1", "strongest_jev_disagreement": "q1"
    }
    from jev_rank.models import MethodResult
    records = top_document_records(examples, {name: MethodResult(name, values) for name, values in rankings.items()})
    assert len(records) == 3
    assert records[0]["documents"][0]["original_bm25_position"] == 1
    assert records[1]["documents"][0]["relevant"] is True

