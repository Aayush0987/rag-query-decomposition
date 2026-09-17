"""
Phase 2 — Teacher label generation.

Samples questions from HotpotQA and uses the Groq API (Llama-3-70B) with a
few-shot prompt to generate decomposed sub-questions for each. Writes
(original_question, decomposed_sub_questions) pairs to a JSONL file.

Usage:
    python src/teacher_labels.py --n 3000 --out data/raw/teacher_labels.jsonl
"""

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from datasets import load_dataset
from groq import Groq, RateLimitError
from tqdm import tqdm

load_dotenv()

# Note: the brief specifies Groq's Llama-3-70B, but that model has since been
# removed from Groq's catalog. Groq's current strong free-tier chat models are
# reasoning models (openai/gpt-oss-20b/120b) whose hidden chain-of-thought
# tokens count against max_tokens and burn through the daily token quota very
# fast for a task this simple. qwen/qwen3.8-27b answers directly with no
# reasoning overhead and is used as the teacher instead (see docs/phase2_notes.md).
TEACHER_MODEL = "qwen/qwen3.8-27b"

FEW_SHOT_PROMPT = """You decompose multi-hop questions into atomic sub-questions.
Each sub-question must be answerable from a single passage, and the sub-questions
together must be sufficient to answer the original question. Output only the
sub-questions, one per line, numbered.

Question: What government position was held by the woman who portrayed Corliss \
Archer in the film Kiss and Tell?
Sub-questions:
1. Who portrayed Corliss Archer in the film Kiss and Tell?
2. What government position did that person hold?

Question: Scott Parkin has been a vocal critic of Exxonmobil and another \
corporation that has operations in how many countries?
Sub-questions:
1. What other corporation has Scott Parkin been a vocal critic of, besides Exxonmobil?
2. In how many countries does that corporation have operations?

Question: {question}
Sub-questions:"""


class QuotaExhausted(Exception):
    """Raised when Groq's daily request quota for the model is used up (as
    opposed to a transient per-minute rate limit, which is worth retrying)."""


def decompose(client: Groq, question: str, retries: int = 3) -> list[str]:
    prompt = FEW_SHOT_PROMPT.format(question=question)
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=TEACHER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            text = resp.choices[0].message.content.strip()
            return parse_subquestions(text)
        except RateLimitError as e:
            headers = getattr(e.response, "headers", {}) or {}
            remaining = headers.get("x-ratelimit-remaining-requests")
            reset = headers.get("x-ratelimit-reset-requests", "")
            # A per-minute token/request limit resets in seconds and is worth
            # retrying; the daily request quota resets in hours and is not.
            if remaining == "0" and "h" in reset:
                raise QuotaExhausted(
                    f"Daily request quota exhausted for {TEACHER_MODEL} "
                    f"(resets in {reset})."
                )
            wait = 2 ** attempt
            print(f"  retry {attempt+1}/{retries} after rate limit (sleeping {wait}s)")
            time.sleep(wait)
        except Exception as e:
            wait = 2 ** attempt
            print(f"  retry {attempt+1}/{retries} after error: {e} (sleeping {wait}s)")
            time.sleep(wait)
    return []


def parse_subquestions(text: str) -> list[str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    subqs = []
    for line in lines:
        cleaned = line.lstrip("0123456789.-) ").strip()
        if cleaned:
            subqs.append(cleaned)
    return subqs


def load_done_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    ids = set()
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.add(json.loads(line)["id"])
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3000, help="target total number of labeled examples")
    parser.add_argument("--out", type=str, default="data/raw/teacher_labels.jsonl")
    parser.add_argument("--sleep", type=float, default=0.3, help="seconds between calls (rate limiting)")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY not set. Copy .env.example to .env and add your key.")

    client = Groq(api_key=api_key)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(out_path)
    if done_ids:
        print(f"Resuming: {len(done_ids)} examples already in {out_path}")

    remaining_target = max(args.n - len(done_ids), 0)
    if remaining_target == 0:
        print(f"Target of {args.n} already reached.")
        return

    print("Loading HotpotQA...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    ds = ds.filter(lambda row: row["id"] not in done_ids)
    ds = ds.select(range(min(remaining_target, len(ds))))

    written = 0
    stopped_early = False
    with open(out_path, "a") as f:
        for row in tqdm(ds, desc="Generating teacher labels"):
            question = row["question"]
            try:
                subqs = decompose(client, question)
            except QuotaExhausted as e:
                print(f"\nStopping: {e}")
                stopped_early = True
                break
            if not subqs:
                continue
            record = {
                "id": row["id"],
                "question": question,
                "sub_questions": subqs,
                "answer": row["answer"],
                "supporting_facts": row["supporting_facts"],
                "context": row["context"],
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            written += 1
            time.sleep(args.sleep)

    total = len(done_ids) + written
    print(f"Wrote {written} new examples this run ({total}/{args.n} total in {out_path})")
    if stopped_early:
        print("Re-run this same command later (once the quota resets) to continue.")


if __name__ == "__main__":
    main()
