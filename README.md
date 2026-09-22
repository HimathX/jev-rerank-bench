# Jev NanoBEIR reranking benchmark

This project compares the supplied BM25 ranking, two local cross-encoders, and two zero-shot [TypeSafe AI Jev](https://docs.typesafe.ai/api) relevance formulations on the NanoNQ split of [NanoBEIR-en](https://huggingface.co/datasets/sentence-transformers/NanoBEIR-en). It is a small pilot benchmark, not definitive evidence of model superiority.

The five methods are:

- the original BM25 order;
- `cross-encoder/ms-marco-MiniLM-L4-v2`;
- `cross-encoder/ms-marco-MiniLM-L6-v2`;
- Jev `noul` binary relevance; and
- Jev `score` graded relevance.

Every reranker receives exactly the same BM25 top-k documents. Missing relevant documents are never injected. Cross-encoder background and evaluation APIs are documented by [Sentence Transformers](https://sbert.net/docs/package_reference/cross_encoder/evaluation.html).

## Installation

Install [uv](https://docs.astral.sh/uv/), then synchronize the locked environment:

```powershell
uv sync
```

Python 3.11 or newer is required. `uv.lock` is committed; do not install this project with a requirements file.

## Commands

A local-only run needs no API key:

```powershell
uv run python -m jev_rank benchmark --dataset nq --rerank-k 20 --at-k 10 --methods bm25,minilm-l4,minilm-l6
```

The two-query smoke test is:

```powershell
uv run python -m jev_rank benchmark --dataset nq --rerank-k 5 --limit-queries 2 --methods bm25,minilm-l4,minilm-l6
```

For the full 50-query pilot, set the key in PowerShell and run:

```powershell
$env:TYPESAFE_API_KEY = "your-key"
uv run python -m jev_rank benchmark --dataset nq --rerank-k 20 --at-k 10 --methods bm25,minilm-l4,minilm-l6,jev-binary,jev-graded --concurrency 16
```

There are 1,000 unique query-document pairs. Because binary and graded relevance are separate requests, a cold-cache full run makes 2,000 Jev API requests, excluding retries. The key is read only from `TYPESAFE_API_KEY`; it is never logged. A missing key fails early only when a Jev method is selected.

Useful options include `--request-timeout`, `--max-retries`, `--jev-model`, `--jev-input-price-per-million`, `--seed`, `--refresh-jev`, `--cache-dir`, `--output-dir`, `--cross-encoder-batch-size`, and `--device`. The `--minilm-l4-model` and `--minilm-l6-model` options accept any `CrossEncoder`-compatible model identifier without changes to evaluation code. Use `--help` for defaults. Device selection is automatic unless `--device` is supplied.

## Data and adding datasets

The benchmark downloads `sentence-transformers/NanoBEIR-en` through Hugging Face `datasets`. Supported aliases are case-sensitive: `nq`, `scifact`, `fiqa2018`, `nfcorpus`, and `hotpotqa`. They map explicitly to the corresponding `NanoNQ`, `NanoSciFact`, `NanoFiQA2018`, `NanoNFCorpus`, and `NanoHotpotQA` splits.

To add another NanoBEIR split, add one explicit alias-to-split entry to `DATASET_SPLITS` in `src/jev_rank/data.py`; unknown aliases are rejected. To expand the experiment from 20 to 100 candidates, pass `--rerank-k 100`. The dataset fingerprint, exact prompts, requested model alias, resolved Jev versions, dependency lock, run date, and runtime configuration are recorded for reproducibility.

## Cache and resume behavior

Successful Jev responses are appended to `artifacts/cache/jev-results.jsonl` before being used. Cache keys cover the dataset repository, fingerprint, split, IDs, method, requested model, and a canonical serialization of the exact payload. On restart, successful records are reused and failed records are retried. A truncated last line is tolerated after interruption. `--refresh-jev` deliberately bypasses hits.

The cache stores query/document content and must not be committed. Writes are serialized and flushed. Authentication and validation failures are not retried indefinitely; transient HTTP 429, 500, 502, 503, 504, and 529 responses use exponential backoff with jitter and honor `Retry-After`. A mixed resolved-version cache is visibly warned about in the report.

## Metrics and cost

One shared implementation computes nDCG@10, MRR@10, MAP@rerank-k, and Recall@10. MAP uses every known relevant qrel in its denominator, so positives absent from the candidates reduce MAP and Recall. Candidate Recall@rerank-k is reported as the shared retrieval ceiling. An evaluation-only qrel oracle puts known relevant candidates first; it is an unattainable diagnostic, not a runnable ranker.

The default Jev input price is configurable and recorded (`$0.042` per million input tokens at the time this benchmark was specified). Estimated cost is:

```text
input_tokens / 1,000,000 * configured_input_price_per_million
```

Output tokens are recorded but excluded while documented output pricing is free. Cost per pair divides by successful uncached Jev pairs only. Cached end-to-end time is separate from live API latency; live p50/p95 and throughput do not include cache hits. Concurrency is an operational setting, not a claim of linear scaling.

## Outputs

Every invocation receives a unique `artifacts/runs/<run-id>/` directory containing:

- `config.json`
- `metrics.json` and `metrics.csv`
- `pair_scores.jsonl`
- `top_documents.jsonl`
- `error_analysis.json`
- `retrieval_ceiling.png`
- `benchmark_report.md`

Only a complete invocation is promoted to `artifacts/results/`. If a Jev pair exhausts its retries, completed cache records and diagnostics remain resumable, partial Jev metrics are omitted, and the run is not promoted.

The report includes every query/method top-three comparison and three deterministic diagnostics based on the predeclared primary Jev graded method: largest improvement, largest regression, and strongest binary-versus-graded ordering disagreement. These selected cases are not representative statistical evidence.

## Tests

```powershell
uv run pytest
```

The client tests mock HTTP and never make paid calls.

## Relevance benchmark versus policy steering

This project measures general relevance reranking on NanoNQ. The supplied Colab notebook instead demonstrates metadata-aware business-policy steering with fields such as source, version, and year. Those are different experimental questions.

Future work could build a separate, metadata-aware policy experiment comparing “prefer official, current documentation” with “prefer an immediate actionable workaround.” NanoNQ lacks the required source/version metadata, so that work needs a purpose-built dataset and separate ground-truth judgments. It is outside this single benchmark run and must not trigger additional Jev calls here.

## Limitations

NanoNQ has only 50 pilot queries and binary, potentially incomplete relevance judgments. Do not infer statistical significance or probability calibration from it. The qrel oracle is bounded by those judgments, and qualitative examples cannot establish general superiority. Jev throughput and latency depend on rate limits, service load, network conditions, and local scheduling.
