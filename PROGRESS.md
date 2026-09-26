# Progress Tracker

Status per phase from `build-brief-query-decomposition.md`. Results and analysis live in [README.md](README.md).

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Study notes | Done (docs/phase0_study_notes.md) |
| 1 | Environment setup, quantized Llama-3-8B, sanity check | Done (docs/phase1_sanity_check.md) |
| 2 | Teacher labels | Done at 1,338 labels (target was 2,000; Groq free-tier quota). Teacher is qwen/qwen3.8-27b since Llama-3-70B was removed from Groq. See docs/phase2_notes.md |
| 3 | Baseline evaluation | Done: raw, base-8B, teacher (with and without the original question) |
| 4 | Data formatting | Done: 1,070 / 133 / 135 split (src/format_data.py) |
| 5 | LoRA fine-tuning | Done: rank 16, 600 iters, best checkpoint at iter 300 by validation loss (docs/training_log.txt) |
| 6 | Post-fine-tune evaluation | Done: results table in README |
| 7 | Integration demo | Done: src/rag_pipeline.py, sample in docs/rag_demo_output.txt |
| 8 | Deployment | Done: fused model, FastAPI /decompose, latency/cost in docs/latency_cost.json |
| 9 | Documentation | Done: README with results, hyperparameters, failure analysis, lessons |
| 7.5 | Stretch: HyDE | Not started |

## Known limitations
- Test set is 135 questions; differences of a few points are within noise.
- Labels: 1,338 rather than the 2,000+ target. `src/teacher_labels.py` is resumable to add more.
- Teacher decomposition alone does not beat the raw query; the gain appears only when the original question is searched alongside the sub-questions.
