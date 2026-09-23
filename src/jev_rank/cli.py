from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import ALL_METHODS, DEFAULT_METHODS, BenchmarkConfig, run_benchmark
from .cross_encoder import DEFAULT_MODELS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jev_rank", description="Benchmark Jev and local cross-encoders on NanoBEIR")
    subparsers = parser.add_subparsers(dest="command", required=True)
    benchmark = subparsers.add_parser("benchmark", help="run the benchmark")
    benchmark.add_argument("--dataset", default="nq")
    benchmark.add_argument("--rerank-k", type=int, default=20)
    benchmark.add_argument("--at-k", type=int, default=10)
    benchmark.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    benchmark.add_argument("--limit-queries", type=int)
    benchmark.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    benchmark.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    benchmark.add_argument("--request-timeout", type=float, default=60.0)
    benchmark.add_argument("--max-retries", type=int, default=5)
    benchmark.add_argument("--concurrency", type=int, default=16)
    benchmark.add_argument("--jev-model", default="jev-latest")
    benchmark.add_argument("--jev-input-price-per-million", type=float, default=0.042)
    benchmark.add_argument("--openai-model", default="gpt-5.4-nano-2026-03-17")
    benchmark.add_argument("--gemini-model", default="gemini-3.5-flash-lite")
    benchmark.add_argument("--openai-input-price-per-million", type=float, default=0.20)
    benchmark.add_argument("--openai-output-price-per-million", type=float, default=1.25)
    benchmark.add_argument("--gemini-input-price-per-million", type=float, default=0.30)
    benchmark.add_argument("--gemini-output-price-per-million", type=float, default=2.50)
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument("--refresh-jev", action="store_true")
    benchmark.add_argument("--refresh-provider", action="store_true")
    benchmark.add_argument("--cross-encoder-batch-size", type=int, default=32)
    benchmark.add_argument("--device", default=None, help="CrossEncoder device, e.g. cpu or cuda; default is automatic")
    benchmark.add_argument("--minilm-l4-model", default=DEFAULT_MODELS["minilm-l4"], help="CrossEncoder-compatible model used by the minilm-l4 slot")
    benchmark.add_argument("--minilm-l6-model", default=DEFAULT_MODELS["minilm-l6"], help="CrossEncoder-compatible model used by the minilm-l6 slot")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "benchmark":
        methods = tuple(part.strip() for part in args.methods.split(",") if part.strip())
        config = BenchmarkConfig(
            dataset=args.dataset, rerank_k=args.rerank_k, at_k=args.at_k, methods=methods,
            limit_queries=args.limit_queries, cache_dir=args.cache_dir, output_dir=args.output_dir,
            request_timeout=args.request_timeout, max_retries=args.max_retries, concurrency=args.concurrency,
            jev_model=args.jev_model, jev_input_price_per_million=args.jev_input_price_per_million,
            openai_model=args.openai_model, gemini_model=args.gemini_model,
            openai_input_price_per_million=args.openai_input_price_per_million,
            openai_output_price_per_million=args.openai_output_price_per_million,
            gemini_input_price_per_million=args.gemini_input_price_per_million,
            gemini_output_price_per_million=args.gemini_output_price_per_million,
            seed=args.seed, refresh_jev=args.refresh_jev, cross_encoder_batch_size=args.cross_encoder_batch_size,
            refresh_provider=args.refresh_provider,
            device=args.device,
            cross_encoder_models={"minilm-l4": args.minilm_l4_model, "minilm-l6": args.minilm_l6_model},
        )
        try:
            run_dir = run_benchmark(config)
        except (ValueError, RuntimeError) as exc:
            raise SystemExit(f"error: {exc}") from exc
        print(f"Benchmark outputs: {run_dir}")
