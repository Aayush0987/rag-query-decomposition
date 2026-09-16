# Progress Tracker

Tracks phase-by-phase status from `build-brief-query-decomposition.md`. Update this
and commit at the end of every work session (see "Daily commit plan" below).

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Study (LoRA, quantization, MLX, query rewriting, distillation, Recall@k/MRR, catastrophic forgetting) | Done |
| 1 | Environment setup (mlx, mlx-lm, quantized Llama-3-8B, sanity check) | Not started |
| 2 | Teacher label generation (Groq Llama-3-70B decompositions on HotpotQA) | Not started |
| 3 | Baseline evaluation (Recall@k for raw / teacher / base-8B) | Not started |
| 4 | Data formatting (MLX JSONL, train/val/test split) | Not started |
| 5 | LoRA fine-tuning | Not started |
| 6 | Post-fine-tune evaluation (Recall@k comparison table) | Not started |
| 7 | Integration demo (end-to-end RAG pipeline) | Not started |
| 8 | Deployment (fused model + FastAPI `/decompose`, latency/cost comparison) | Not started |
| 9 | Documentation (README with results, diagrams, failure analysis) | Not started |
| 7.5 | Stretch: HyDE | Not started |

## Daily commit plan

To keep a visible commit streak without hollow commits, each session should end with
at least one real commit. Natural stopping points per phase, roughly one or more
commits per session:

- Day 1: repo scaffold + Phase 0 notes (this commit)
- Day 2: Phase 1 — MLX installed, model quantized/downloaded, sanity-check script + output
- Day 3: Phase 2 — teacher label generation script, first batch of labels committed (data sample, not full raw dump if large)
- Day 4: Phase 3 — FAISS baseline retrieval pipeline + Recall@k numbers for raw/teacher/base
- Day 5: Phase 4 — data formatting script + JSONL splits (small enough to commit; else commit generation script + stats)
- Day 6-7: Phase 5 — LoRA training runs, hyperparameter log, checkpoint selection notes (checkpoints themselves gitignored — log metrics in `docs/`)
- Day 8: Phase 6 — post-fine-tune Recall@k, comparison table into README
- Day 9: Phase 7 — end-to-end RAG demo script
- Day 10: Phase 8 — fused model + FastAPI endpoint + latency/cost table
- Day 11: Phase 9 — full README writeup
- Day 12+: Phase 7.5 stretch (HyDE)

Long-running steps (label generation batches, LoRA training) can span multiple
sessions — commit intermediate progress (partial label files, in-progress training
logs) rather than waiting for the phase to fully close, so a slow phase doesn't
create a commit gap.
