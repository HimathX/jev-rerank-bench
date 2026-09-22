from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .models import Candidate, QueryExample

DATASET_REPOSITORY = "sentence-transformers/NanoBEIR-en"
DATASET_SPLITS = {
    "nq": "NanoNQ",
    "scifact": "NanoSciFact",
    "fiqa2018": "NanoFiQA2018",
    "nfcorpus": "NanoNFCorpus",
    "hotpotqa": "NanoHotpotQA",
}


@dataclass(frozen=True)
class BenchmarkData:
    examples: tuple[QueryExample, ...]
    repository: str
    split: str
    fingerprint: str


def resolve_split(alias: str) -> str:
    if alias not in DATASET_SPLITS:
        valid = ", ".join(DATASET_SPLITS)
        raise ValueError(f"Unknown dataset alias {alias!r}. Use one of: {valid} (aliases are case-sensitive).")
    return DATASET_SPLITS[alias]


def _field(row: Mapping[str, Any], names: Sequence[str], label: str) -> Any:
    present = [name for name in names if name in row]
    if not present:
        raise ValueError(f"Missing required {label}; expected one of {list(names)}, got {sorted(row)}")
    return row[present[0]]


def transform_rows(
    corpus_rows: Iterable[Mapping[str, Any]],
    query_rows: Iterable[Mapping[str, Any]],
    qrel_rows: Iterable[Mapping[str, Any]],
    bm25_rows: Iterable[Mapping[str, Any]],
    rerank_k: int,
    limit_queries: int | None = None,
) -> tuple[QueryExample, ...]:
    corpus: dict[str, str] = {}
    for row in corpus_rows:
        doc_id = str(_field(row, ("_id", "id", "corpus-id"), "corpus id"))
        text = str(_field(row, ("text",), "corpus text"))
        corpus[doc_id] = text

    queries: dict[str, str] = {}
    for row in query_rows:
        query_id = str(_field(row, ("_id", "id", "query-id"), "query id"))
        queries[query_id] = str(_field(row, ("text",), "query text"))

    qrels: dict[str, set[str]] = {query_id: set() for query_id in queries}
    for row in qrel_rows:
        query_id = str(_field(row, ("query-id", "query_id"), "qrel query id"))
        document_id = str(_field(row, ("corpus-id", "corpus_id", "document-id"), "qrel corpus id"))
        qrels.setdefault(query_id, set()).add(document_id)

    bm25: dict[str, list[str]] = {}
    for row in bm25_rows:
        query_id = str(_field(row, ("query-id", "query_id"), "BM25 query id"))
        ids = _field(row, ("corpus-ids", "corpus_ids"), "BM25 corpus ids")
        if not isinstance(ids, (list, tuple)):
            raise ValueError(f"BM25 corpus ids for query {query_id!r} must be a list")
        bm25[query_id] = [str(value) for value in ids]

    missing = [query_id for query_id in queries if query_id not in bm25]
    if missing:
        raise ValueError(f"BM25 rows missing for {len(missing)} queries; first missing id: {missing[0]}")

    examples: list[QueryExample] = []
    for query_id, query_text in queries.items():
        candidate_ids = bm25[query_id][:rerank_k]
        absent = [document_id for document_id in candidate_ids if document_id not in corpus]
        if absent:
            raise ValueError(f"BM25 references corpus id {absent[0]!r} absent from the corpus")
        candidates = tuple(
            Candidate(document_id, corpus[document_id], position)
            for position, document_id in enumerate(candidate_ids, start=1)
        )
        examples.append(QueryExample(query_id, query_text, candidates, frozenset(qrels.get(query_id, set()))))
        if limit_queries is not None and len(examples) >= limit_queries:
            break
    return tuple(examples)


def load_nanobeir(alias: str, rerank_k: int, limit_queries: int | None = None) -> BenchmarkData:
    from datasets import load_dataset

    split = resolve_split(alias)
    subsets = {
        name: load_dataset(DATASET_REPOSITORY, name, split=split)
        for name in ("corpus", "queries", "qrels", "bm25")
    }
    fingerprints = [str(getattr(dataset, "_fingerprint", "unknown")) for dataset in subsets.values()]
    fingerprint = ":".join(fingerprints)
    examples = transform_rows(
        subsets["corpus"], subsets["queries"], subsets["qrels"], subsets["bm25"], rerank_k, limit_queries
    )
    return BenchmarkData(examples, DATASET_REPOSITORY, split, fingerprint)

