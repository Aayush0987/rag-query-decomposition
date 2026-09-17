"""
Phase 3 / Phase 6 — Retrieval evaluation.

Builds a FAISS index over HotpotQA's own context paragraphs and measures
Recall@k for a given query strategy (raw question, teacher decomposition,
model-generated decomposition). Sub-question results are merged (union,
deduplicated) before computing recall for decomposition strategies.

Usage:
    python src/eval_retrieval.py --strategy raw --k 5 10 --n 500
    python src/eval_retrieval.py --strategy teacher --labels data/raw/teacher_labels.jsonl --k 5 10
    python src/eval_retrieval.py --strategy model --model-path <mlx model or adapter path> --k 5 10 --n 500
"""

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_corpus(dataset):
    """Flatten each example's context into (title, sentences) paragraphs and
    return a global list of paragraph texts + an index mapping example id ->
    the set of global paragraph indices that came from that example (needed
    because HotpotQA's `context` field is per-question, not a shared corpus,
    so we build one corpus per evaluated example to keep this self-contained
    and tractable on a laptop)."""
    paragraphs = []
    titles = []
    for title, sentences in zip(dataset["title"], dataset["sentences"]):
        text = " ".join(sentences)
        paragraphs.append(text)
        titles.append(title)
    return paragraphs, titles


def gold_titles(supporting_facts):
    return set(supporting_facts["title"])


def recall_at_k(retrieved_titles: list[str], gold: set[str], k: int) -> float:
    top_k = set(retrieved_titles[:k])
    if not gold:
        return 0.0
    return len(top_k & gold) / len(gold)


def embed(model, texts):
    return np.asarray(model.encode(texts, show_progress_bar=False, normalize_embeddings=True))


def retrieve(model, index, titles, queries: list[str], k: int) -> list[str]:
    """Retrieve top-k paragraph titles for a list of queries (sub-questions),
    merged and deduplicated preserving first-seen order."""
    q_emb = embed(model, queries)
    _, idxs = index.search(q_emb, k)
    seen = []
    for row in idxs:
        for i in row:
            t = titles[i]
            if t not in seen:
                seen.append(t)
    return seen


def load_teacher_labels(path):
    labels = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            labels[rec["id"]] = rec["sub_questions"]
    return labels


def model_decompose(mlx_model_path, question):
    from mlx_lm import load, generate

    if not hasattr(model_decompose, "_cache"):
        model_decompose._cache = load(mlx_model_path)
    model, tokenizer = model_decompose._cache

    prompt = (
        "Decompose this multi-hop question into 2-3 atomic sub-questions, "
        "one per line, numbered. Output only the sub-questions.\n\n"
        f"Question: {question}\nSub-questions:"
    )
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    text = generate(model, tokenizer, prompt=formatted, max_tokens=150, verbose=False)
    lines = [l.strip().lstrip("0123456789.-) ").strip() for l in text.splitlines() if l.strip()]
    return [l for l in lines if l] or [question]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["raw", "teacher", "model"], required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--n", type=int, default=500, help="number of eval questions")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--labels", type=str, default="data/raw/teacher_labels.jsonl")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--out", type=str, default=None, help="optional path to append JSON result line")
    args = parser.parse_args()

    print(f"Loading HotpotQA [{args.split}]...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=args.split)
    ds = ds.select(range(min(args.n, len(ds))))

    embedder = SentenceTransformer(EMBED_MODEL)

    teacher_labels = load_teacher_labels(args.labels) if args.strategy == "teacher" else None

    max_k = max(args.k)
    scores = {k: [] for k in args.k}

    for row in tqdm(ds, desc=f"Evaluating strategy={args.strategy}"):
        paragraphs, titles = build_corpus(row["context"])
        p_emb = embed(embedder, paragraphs)
        index = faiss.IndexFlatIP(p_emb.shape[1])
        index.add(p_emb)

        gold = gold_titles(row["supporting_facts"])

        if args.strategy == "raw":
            queries = [row["question"]]
        elif args.strategy == "teacher":
            queries = teacher_labels.get(row["id"], [row["question"]])
        else:
            queries = model_decompose(args.model_path, row["question"])

        retrieved = retrieve(embedder, index, titles, queries, max_k)

        for k in args.k:
            scores[k].append(recall_at_k(retrieved, gold, k))

    result = {"strategy": args.strategy, "n": len(ds)}
    for k in args.k:
        mean_recall = float(np.mean(scores[k]))
        result[f"recall@{k}"] = mean_recall
        print(f"Recall@{k}: {mean_recall:.4f}")

    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
