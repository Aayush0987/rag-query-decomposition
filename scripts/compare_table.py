"""Render docs/recall_results.jsonl as the four-row Recall@k comparison table."""
import json
import sys

ROWS = [("raw", "Raw query (floor)"), ("base", "Base Llama-3-8B, zero-shot"),
        ("finetuned", "Fine-tuned Llama-3-8B (LoRA)"), ("teacher", "Teacher (ceiling)")]

latest = {}
for line in open(sys.argv[1] if len(sys.argv) > 1 else "docs/recall_results.jsonl"):
    r = json.loads(line)
    latest[r.get("label", r["strategy"])] = r

print("| Approach | n | Recall@5 | Recall@10 |\n|---|---|---|---|")
for key, name in ROWS:
    if key in latest:
        r = latest[key]
        print(f"| {name} | {r['n']} | {r['recall@5']:.4f} | {r['recall@10']:.4f} |")
