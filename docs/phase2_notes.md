# Phase 2 — Teacher Label Generation: Notes

## Deviations from the build brief

The brief specifies Groq's Llama-3-70B as the teacher model. That model has
since been removed from Groq's catalog entirely (`llama-3.3-70b-versatile`,
`llama-3.1-8b-instant`, `llama3-70b-8192`, `mixtral-8x7b-32768`, `gemma2-9b-it`
all return 404/decommissioned errors as of this project). Groq's currently
available strong models are:

- `openai/gpt-oss-20b` / `openai/gpt-oss-120b` — reasoning models. Their hidden
  chain-of-thought tokens count against `max_tokens` and consumed the entire
  daily free-tier token quota (200k tokens/day) after only ~100 calls, with
  many calls truncating (`finish_reason: length`) before producing any visible
  answer.
- `qwen/qwen3.8-27b` — answers directly with no reasoning-token overhead
  (`reasoning` field is `None`), ~30-40 completion tokens per decomposition
  call, `finish_reason: stop` on all 300 validation calls.

**Teacher model used: `qwen/qwen3.8-27b`** (see `src/teacher_labels.py`).

## Validation run

- 300 questions from HotpotQA train split
- 300/300 produced non-empty decompositions (0 failures/retries)
- 265/300 produced 2+ sub-questions, 35/300 produced only 1 (acceptable —
  some HotpotQA questions are close to single-hop even though labeled
  multi-hop)
- Output: `data/raw/teacher_labels_sample.jsonl` (not committed in full to
  keep repo size down — a small excerpt may be kept for reference; the full
  label set for training lives in `data/raw/teacher_labels.jsonl`, gitignored)

## Also fixed: dataset repo id

`hotpot_qa` (the canonical, non-namespaced dataset id) no longer resolves with
current `datasets` versions — HF now requires namespaced ids for legacy
"canonical" datasets. Using `hotpotqa/hotpot_qa` instead (same schema:
`id`, `question`, `answer`, `type`, `level`, `supporting_facts`, `context`).
