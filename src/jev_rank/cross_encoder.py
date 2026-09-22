from __future__ import annotations

import time
from collections.abc import Sequence

from .evaluation import rank_scores
from .models import MethodResult, QueryExample

DEFAULT_MODELS = {
    "minilm-l4": "cross-encoder/ms-marco-MiniLM-L4-v2",
    "minilm-l6": "cross-encoder/ms-marco-MiniLM-L6-v2",
}


def run_cross_encoder(
    method: str,
    model_id: str,
    examples: Sequence[QueryExample],
    batch_size: int,
    device: str | None,
) -> MethodResult:
    from sentence_transformers.cross_encoder import CrossEncoder

    started = time.perf_counter()
    model = CrossEncoder(model_id, device=device)
    pairs = [(example.text, candidate.text) for example in examples for candidate in example.candidates]
    inference_started = time.perf_counter()
    raw_scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=True)
    inference_seconds = time.perf_counter() - inference_started
    scores = [float(value) for value in raw_scores]
    rankings = {}
    offset = 0
    for example in examples:
        count = len(example.candidates)
        chunk = scores[offset : offset + count]
        rankings[example.query_id] = rank_scores(chunk, [c.document_id for c in example.candidates])
        offset += count
    per_pair = inference_seconds / len(pairs) if pairs else 0.0
    result = MethodResult(method, rankings, [per_pair] * len(pairs), time.perf_counter() - started)
    result.successful_uncached_pairs = len(pairs)
    result.live_call_seconds = inference_seconds
    return result
