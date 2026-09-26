"""Per-question comparison of raw vs fine-tuned decomposition (Recall@5), from cached outputs."""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "src")
import faiss
from sentence_transformers import SentenceTransformer

from eval_retrieval import EMBED_MODEL, build_shared_corpus, embed, gold_titles, recall_at_k, retrieve

recs = [json.loads(l) for l in open("data/processed/test_full.jsonl")]
cache = {}
for l in open("data/processed/decomp_cache.jsonl"):
    r = json.loads(l)
    if r["adapter"] == "adapters/best":
        cache[r["question"]] = r["sub_questions"]

emb = SentenceTransformer(EMBED_MODEL)
paras, titles = build_shared_corpus(recs)
index = faiss.IndexFlatIP(384)
index.add(embed(emb, paras))

rows = []
for r in recs:
    gold = gold_titles(r["supporting_facts"])
    subs = cache[r["question"]]
    q = r["question"]
    rows.append({
        "q": q, "n_sub": len(subs), "subs": subs, "teacher_subs": r["sub_questions"],
        "raw": recall_at_k(retrieve(emb, index, titles, [q], 5), gold, 5),
        "ft": recall_at_k(retrieve(emb, index, titles, subs, 5), gold, 5),
        "ft_orig": recall_at_k(retrieve(emb, index, titles, [q] + subs, 5), gold, 5),
    })

def mean(xs): return sum(xs) / len(xs) if xs else float("nan")
print(f"n={len(rows)}")
for k in (1, 2, 3):
    g = [r for r in rows if (r["n_sub"] == k if k < 3 else r["n_sub"] >= 3)]
    print(f"{k}{'+' if k == 3 else ''} sub-questions: n={len(g)} raw={mean([r['raw'] for r in g]):.3f} "
          f"ft={mean([r['ft'] for r in g]):.3f} ft+orig={mean([r['ft_orig'] for r in g]):.3f}")
better = [r for r in rows if r["ft"] > r["raw"]]
worse = [r for r in rows if r["ft"] < r["raw"]]
print(f"ft-only vs raw: better on {len(better)}, worse on {len(worse)}, tied on {len(rows)-len(better)-len(worse)}")
print("\nWORSE than raw (examples):")
for r in worse[:6]:
    print(f"- {r['q']}\n    ft: {r['subs']}  (raw={r['raw']:.1f} ft={r['ft']:.1f})")
print("\nBETTER than raw (examples):")
for r in better[:4]:
    print(f"- {r['q']}\n    ft: {r['subs']}  (raw={r['raw']:.1f} ft={r['ft']:.1f})")
