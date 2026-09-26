"""
Phase 3 / Phase 6 — Retrieval evaluation.

Builds a FAISS index over HotpotQA's own context paragraphs and measures
Recall@k for a given query strategy (raw question, teacher decomposition,
model-generated decomposition). Sub-question results are merged (union,
deduplicated) before computing recall for decomposition strategies.

Usage:
    python src/eval_retrieval.py --strategy raw --k 5 10
    python src/eval_retrieval.py --strategy teacher --k 5 10
    python src/eval_retrieval.py --strategy model --model-path <mlx model> [--adapter-path <adapters dir>]

Evaluates on the held-out test split written by src/format_data.py.
"""

import argparse
import json
import os
from pathlib import Path

# FAISS and torch each bundle OpenMP on macOS; multi-query search segfaults without this.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

from prompts import build_prompt, parse_subquestions

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_shared_corpus(ds):
    """Pool every context paragraph across the whole sampled eval set into one
    shared corpus (deduplicated by title), so retrieval is a real search
    problem — each question's gold passages sit among thousands of distractor
    paragraphs pulled from every OTHER question's context, not just its own
    ~10-paragraph distractor set. Building one tiny per-question index (the
    previous approach) made Recall@10 trivially 1.0 whenever the corpus itself
    only had ~10 paragraphs."""
    paragraphs = []
    titles = []
    seen_titles = set()
    for row in ds:
        for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
            if title in seen_titles:
                continue
            seen_titles.add(title)
            paragraphs.append(" ".join(sentences))
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
    """Retrieve top-k paragraph titles per query (sub-question), merged by
    interleaving ranks (every query's rank-1 hit, then every rank-2 hit, ...)
    and deduplicated, so truncating to the first k gives each sub-question a
    fair share rather than favouring the first one."""
    q_emb = embed(model, queries)
    _, idxs = index.search(q_emb, k)
    seen = []
    for rank in range(k):
        for row in idxs:
            t = titles[row[rank]]
            if t not in seen:
                seen.append(t)
    return seen


CACHE_FILE = "data/processed/decomp_cache.jsonl"


def model_decompose(mlx_model_path, question, adapter_path=None):
    """Decompose with a local MLX model; results are cached on disk keyed by
    (model, adapter, question) so re-running variants doesn't regenerate."""
    if not hasattr(model_decompose, "_cache"):
        cache = {}
        if os.path.exists(CACHE_FILE):
            for line in open(CACHE_FILE):
                r = json.loads(line)
                cache[(r["model"], r["adapter"], r["question"])] = r["sub_questions"]
        model_decompose._cache = cache
        model_decompose._model = None
    key = (mlx_model_path, adapter_path, question)
    if key in model_decompose._cache:
        return model_decompose._cache[key]

    if model_decompose._model is None:
        from decomposer import LocalDecomposer
        model_decompose._model = LocalDecomposer(mlx_model_path, adapter_path)
    subqs = model_decompose._model.decompose(question)
    model_decompose._cache[key] = subqs
    with open(CACHE_FILE, "a") as f:
        f.write(json.dumps({"model": mlx_model_path, "adapter": adapter_path,
                            "question": question, "sub_questions": subqs}) + "\n")
    return subqs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["raw", "teacher", "model"], required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--n", type=int, default=500, help="number of eval questions")
    parser.add_argument("--eval-file", type=str, default="data/processed/test_full.jsonl",
                        help="held-out labeled records (from format_data.py) to evaluate on")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--include-original", action="store_true",
                        help="also search with the original question alongside the sub-questions")
    parser.add_argument("--label", type=str, default=None, help="row name in results (raw/base/finetuned/teacher)")
    parser.add_argument("--out", type=str, default=None, help="optional path to append JSON result line")
    args = parser.parse_args()

    with open(args.eval_file) as f:
        ds = [json.loads(l) for l in f if l.strip()][: args.n]

    embedder = SentenceTransformer(EMBED_MODEL)

    teacher_labels = {r["id"]: r["sub_questions"] for r in ds} if args.strategy == "teacher" else None

    print("Building shared corpus across all sampled questions...")
    paragraphs, titles = build_shared_corpus(ds)
    print(f"Shared corpus: {len(paragraphs)} unique paragraphs")
    p_emb = embed(embedder, paragraphs)
    index = faiss.IndexFlatIP(p_emb.shape[1])
    index.add(p_emb)

    max_k = max(args.k)
    scores = {k: [] for k in args.k}

    for row in tqdm(ds, desc=f"Evaluating strategy={args.strategy}"):
        gold = gold_titles(row["supporting_facts"])

        if args.strategy == "raw":
            queries = [row["question"]]
        elif args.strategy == "teacher":
            queries = teacher_labels.get(row["id"], [row["question"]])
        else:
            queries = model_decompose(args.model_path, row["question"], args.adapter_path)

        if args.include_original and args.strategy != "raw":
            queries = [row["question"]] + queries

        retrieved = retrieve(embedder, index, titles, queries, max_k)

        for k in args.k:
            scores[k].append(recall_at_k(retrieved, gold, k))

    result = {"strategy": args.strategy, "label": args.label or args.strategy, "n": len(ds)}
    for k in args.k:
        mean_recall = float(np.mean(scores[k]))
        result[f"recall@{k}"] = mean_recall
        print(f"Recall@{k}: {mean_recall:.4f}")

    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
