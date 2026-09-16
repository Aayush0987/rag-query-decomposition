# Build Brief: RAG Query Decomposition via Distilled LoRA Fine-Tuning

## Project Summary
Distill a strong teacher model's query-decomposition ability into a small, locally
fine-tuned Llama-3-8B model, to improve multi-hop retrieval in RAG pipelines without
per-query API cost or latency. Fine-tuning is done via LoRA using Apple's MLX framework
on Apple Silicon (M4 Pro, 24GB unified memory).

## Core Hypothesis
A fine-tuned 8B model can approximate a much larger teacher model's query-decomposition
quality, closing most of the retrieval-accuracy gap between "no rewriting" and
"frontier-model rewriting" — at zero marginal API cost.

## Tech Stack
- **Base model**: Llama-3-8B (4-bit quantized)
- **Fine-tuning**: MLX + `mlx-lm` (LoRA)
- **Teacher model**: Groq API (Llama-3-70B) — free tier, used only to generate training labels
- **Dataset**: HotpotQA (multi-hop QA, includes gold supporting-fact context paragraphs)
- **Retrieval**: FAISS, using HotpotQA's own context paragraphs as the corpus
- **Serving**: FastAPI
- **Environment**: macOS, Apple M4 Pro, 24GB RAM

---

## Phase 0 — Study (before coding)
- LoRA / QLoRA mechanics
- Quantization (4-bit) tradeoffs
- MLX framework basics (vs. PyTorch)
- Query rewriting techniques: expansion, decomposition, HyDE, step-back prompting
- Knowledge distillation (teacher → student)
- Retrieval metrics: Recall@k, MRR
- Catastrophic forgetting in fine-tuning

## Phase 1 — Environment Setup
- Install `mlx`, `mlx-lm`
- Download and quantize Llama-3-8B (4-bit) via MLX
- Sanity check: run base model, confirm it generates coherent text locally

## Phase 2 — Teacher Label Generation (Distillation Data)
- Sample ~2,000–5,000 questions from HotpotQA
- Use Groq API (Llama-3-70B) with a few-shot prompt to generate decomposed
  sub-questions for each original question
- Few-shot examples in the prompt should show: original multi-hop question →
  2 (sometimes more) atomic sub-questions that together answer it
- Store as (original_question, decomposed_sub_questions) pairs
- Rate-limit/batch calls appropriately for Groq's free tier

## Phase 3 — Baseline Evaluation (before fine-tuning)
Build a minimal retrieval pipeline:
- Index HotpotQA's own context paragraphs (the corpus) into FAISS
- Measure **Recall@k** (k = 5 and 10) using three query strategies:
  1. Raw original question — no rewriting (floor baseline)
  2. Teacher (Groq/Llama-3-70B) decomposition — ceiling baseline
  3. Base Llama-3-8B, zero-shot prompted to decompose — middle baseline
- Record all three numbers before any fine-tuning happens

## Phase 4 — Data Formatting
- Convert teacher-generated pairs into MLX's expected JSONL format
  (prompt = original question, completion = decomposed sub-questions)
- Train/val/test split (e.g., 80/10/10)

## Phase 5 — LoRA Fine-Tuning
- Fine-tune with `mlx_lm.lora`
- Track and log: rank, alpha, learning rate, iterations, training/validation loss
- Save checkpoints; select best checkpoint by validation loss

## Phase 6 — Post-Fine-Tune Evaluation
- Re-run the same Recall@k retrieval test (k=5, k=10), now using the
  **fine-tuned model's** decompositions
- Produce a comparison table with four rows: raw query / base model / fine-tuned
  model / teacher ceiling
- This table is the headline result of the project

## Phase 7 — Integration Demo (self-contained RAG pipeline)
- Build a small end-to-end pipeline: query → decompose (fine-tuned model) →
  retrieve per sub-question from FAISS (HotpotQA corpus) → merge results → generate answer
- Fully self-contained — no dependency on any external/prior project

## Phase 8 — Deployment
- Fuse LoRA adapter into the base model (`mlx_lm.fuse`)
- FastAPI endpoint `/decompose`: input = query, output = sub-questions
- Include a latency + cost comparison: local fine-tuned model vs. calling
  Groq/Llama-3-70B per query — concrete numbers for interview discussion

## Phase 9 — Documentation
README should include:
- Problem statement (why query rewriting matters for RAG)
- Architecture diagram
- Full Recall@k comparison table (all 4 approaches)
- Latency/cost comparison table
- Hyperparameters used
- Failure case analysis — what kinds of questions still don't decompose well
- Lessons learned

---

## Phase 7.5 — Stretch Goal: HyDE (Hypothetical Document Embeddings)

**Only start after Phase 9 core project is complete and working end-to-end.**

- Concept: instead of decomposing the question, generate a hypothetical answer
  to it, and embed that hypothetical answer for retrieval instead of the raw query
- Generate teacher labels for HyDE the same way as Phase 2 (Groq API, few-shot
  prompted to produce a plausible hypothetical answer per question)
- Fine-tune the same base model (either as a second LoRA adapter, or multi-task
  with decomposition — decide based on how Phase 5 went) to generate HyDE-style
  hypothetical answers
- Extend the Recall@k evaluation table with a HyDE row, and optionally a
  "decomposition + HyDE combined" row
- Add a short analysis: which technique works better for which question types
  (e.g., HyDE may help vocabulary-mismatch cases more than multi-hop cases,
  decomposition may help multi-hop cases more)

---

## Success Criteria
- Working baseline vs. fine-tuned comparison with real Recall@k numbers
- Fine-tuned model measurably closes the gap between raw-query and teacher-ceiling performance
- Deployed, callable API endpoint
- Documented, reproducible, interview-ready README
