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
from groq import Groq
from tqdm import tqdm

load_dotenv()

TEACHER_MODEL = "llama-3.3-70b-versatile"

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3000, help="number of questions to sample")
    parser.add_argument("--out", type=str, default="data/raw/teacher_labels.jsonl")
    parser.add_argument("--sleep", type=float, default=0.3, help="seconds between calls (rate limiting)")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY not set. Copy .env.example to .env and add your key.")

    client = Groq(api_key=api_key)

    print("Loading HotpotQA...")
    ds = load_dataset("hotpot_qa", "distractor", split="train")
    n = min(args.n, len(ds))
    ds = ds.select(range(n))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(out_path, "w") as f:
        for row in tqdm(ds, desc="Generating teacher labels"):
            question = row["question"]
            subqs = decompose(client, question)
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
            written += 1
            time.sleep(args.sleep)

    print(f"Wrote {written}/{n} labeled examples to {out_path}")


if __name__ == "__main__":
    main()
