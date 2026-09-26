"""
Phase 4 — Convert teacher labels into MLX-LM LoRA training data.

Writes chat-format JSONL (train/valid/test) to data/processed/, splitting by
question id 80/10/10 with a fixed seed. Also writes test_full.jsonl with the
complete records (context + supporting facts) so retrieval evaluation runs on
questions the student never trained on.

Usage:
    python src/format_data.py --labels data/raw/teacher_labels.jsonl
"""

import argparse
import json
import random
from pathlib import Path

from prompts import build_prompt, format_completion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/raw/teacher_labels.jsonl")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.labels) as f:
        records = [json.loads(l) for l in f if l.strip()]
    records = [r for r in records if r["sub_questions"]]

    rng = random.Random(args.seed)
    rng.shuffle(records)
    n = len(records)
    n_train, n_val = int(n * 0.8), int(n * 0.1)
    splits = {
        "train": records[:n_train],
        "valid": records[n_train:n_train + n_val],
        "test": records[n_train + n_val:],
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, recs in splits.items():
        with open(out / f"{name}.jsonl", "w") as f:
            for r in recs:
                messages = [
                    {"role": "user", "content": build_prompt(r["question"])},
                    {"role": "assistant", "content": format_completion(r["sub_questions"])},
                ]
                f.write(json.dumps({"messages": messages}) + "\n")
        print(f"{name}: {len(recs)}")

    with open(out / "test_full.jsonl", "w") as f:
        for r in splits["test"]:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
