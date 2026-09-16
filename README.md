# RAG Query Decomposition via Distilled LoRA Fine-Tuning

Distilling a large teacher model's (Groq Llama-3-70B) query-decomposition ability
into a small, locally fine-tuned Llama-3-8B (4-bit, MLX + LoRA) to improve
multi-hop retrieval in RAG pipelines — at zero marginal API cost and no
per-query latency hit.

Full build plan: [build-brief-query-decomposition.md](build-brief-query-decomposition.md).
Live phase status: [PROGRESS.md](PROGRESS.md).

## Problem statement

Multi-hop questions (e.g. "What nationality is the director of the film that
won the 2020 award X?") often can't be answered by retrieving passages for the
raw question — no single passage contains the full chain of facts. Decomposing
the question into atomic sub-questions and retrieving separately per
sub-question closes this gap, but doing that decomposition with a frontier
model on every query is expensive and slow. This project tests whether a small
model can be fine-tuned to approximate that decomposition quality locally.

## Architecture

_Diagram to be added in Phase 9 (Phase 7 pipeline: query → decompose → retrieve
per sub-question → merge → generate answer)._

## Tech stack

- Base model: Llama-3-8B, 4-bit quantized
- Fine-tuning: MLX + `mlx-lm` LoRA
- Teacher: Groq API (Llama-3-70B), free tier — training-label generation only
- Dataset: HotpotQA
- Retrieval: FAISS over HotpotQA context paragraphs
- Serving: FastAPI
- Hardware: Apple M4 Pro, 24GB unified memory

## Results

_Recall@k comparison table (raw query / base 8B / fine-tuned 8B / teacher
ceiling) — filled in after Phase 6._

_Latency + cost comparison (local fine-tuned vs. Groq API per query) — filled in
after Phase 8._

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your GROQ_API_KEY
```

## Repo layout

```
docs/           study notes, hyperparameter logs, failure-case analysis
src/            pipeline code (labels, eval, training glue, RAG pipeline, API)
scripts/        one-off / CLI entry points
data/           raw + processed data (gitignored except small samples)
```

## Failure case analysis

_Added in Phase 9._

## Lessons learned

_Added in Phase 9._
