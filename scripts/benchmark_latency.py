"""
Phase 8 — Latency/cost: local fused model vs. Groq teacher per query.

    python scripts/benchmark_latency.py --model models/fused --n 30
Writes docs/latency_cost.json. Pass --price-in/--price-out (USD per 1M tokens)
to add a paid-API cost estimate; the free tier is $0 but capped at 1000 req/day.
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, "src")
from dotenv import load_dotenv
from groq import Groq

from decomposer import LocalDecomposer
from prompts import build_prompt
from teacher_labels import FEW_SHOT_PROMPT, TEACHER_MODEL

load_dotenv()


def summarize(ms):
    ms = sorted(ms)
    return {"mean_ms": round(statistics.mean(ms), 1), "p50_ms": round(ms[len(ms) // 2], 1),
            "p95_ms": round(ms[int(len(ms) * 0.95) - 1], 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--price-in", type=float, default=None)
    ap.add_argument("--price-out", type=float, default=None)
    args = ap.parse_args()

    with open("data/processed/test_full.jsonl") as f:
        questions = [json.loads(l)["question"] for l in f][: args.n]

    local = LocalDecomposer(args.model)
    local.decompose(questions[0])  # warm-up
    local_ms = []
    for q in questions:
        t = time.perf_counter()
        local.decompose(q)
        local_ms.append((time.perf_counter() - t) * 1000)

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    api_ms, tok_in, tok_out = [], 0, 0
    for q in questions:
        t = time.perf_counter()
        r = client.chat.completions.create(
            model=TEACHER_MODEL, temperature=0.0, max_tokens=200,
            messages=[{"role": "user", "content": FEW_SHOT_PROMPT.format(question=q)}])
        api_ms.append((time.perf_counter() - t) * 1000)
        tok_in += r.usage.prompt_tokens
        tok_out += r.usage.completion_tokens
        time.sleep(1.5)

    result = {"n": len(questions), "local_fused_8b": summarize(local_ms),
              "groq_teacher": {"model": TEACHER_MODEL, **summarize(api_ms),
                               "avg_prompt_tokens": round(tok_in / len(questions), 1),
                               "avg_completion_tokens": round(tok_out / len(questions), 1)}}
    if args.price_in is not None and args.price_out is not None:
        cost = (tok_in * args.price_in + tok_out * args.price_out) / 1e6 / len(questions)
        result["groq_teacher"]["est_cost_per_query_usd"] = cost
        result["groq_teacher"]["est_cost_per_1k_queries_usd"] = cost * 1000
    result["local_fused_8b"]["marginal_cost_usd"] = 0.0
    json.dump(result, open("docs/latency_cost.json", "w"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
