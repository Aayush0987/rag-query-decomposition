# Phase 0 — Study Notes

Reference notes for the concepts underpinning this project. Written to be
interview-usable, not exhaustive.

## LoRA / QLoRA

- **LoRA (Low-Rank Adaptation)**: freeze the base model's weights; for targeted
  weight matrices `W` (typically attention projections), learn a low-rank update
  `ΔW = B @ A` where `A` is `r x d` and `B` is `d x r`, `r << d`. Forward pass becomes
  `Wx + BAx`. Only `A`/`B` are trained → orders of magnitude fewer trainable params
  than full fine-tuning, small checkpoint size, easy to swap/merge adapters.
- **Key hyperparameters**: rank `r` (capacity of the update, typically 4-64),
  `alpha` (scaling factor applied as `alpha/r`, controls update magnitude), target
  modules (which weight matrices get adapters — usually q_proj/v_proj or all
  attention+MLP projections), dropout on the LoRA path.
- **QLoRA**: LoRA on top of a *quantized* (typically 4-bit NF4) frozen base model.
  Base weights stay quantized during training; LoRA adapters are still trained in
  higher precision (bf16/fp32). This is what makes fine-tuning an 8B model feasible
  on 24GB unified memory — the frozen weights take ~4-5GB instead of ~16GB.
- **Fusing**: after training, `B @ A` can be added back into `W` to produce a single
  dense fine-tuned model with no inference-time overhead (`mlx_lm.fuse`).

## Quantization (4-bit)

- Reduces weight precision from 16-bit float to ~4 bits per weight, cutting memory
  ~4x with a real but often small quality hit.
- Common schemes: **group-wise quantization** — weights are quantized in small
  groups (e.g. 32/64 values) each with their own scale/zero-point, which keeps
  accuracy much closer to full precision than a single global scale.
- Tradeoff: lower precision → faster load, less RAM, faster memory-bound inference
  on Apple Silicon; but degraded numerical accuracy, especially for very sensitive
  computations (rare in decoder-only LLM inference/fine-tuning).
- MLX represents quantized weights natively and can quantize on load
  (`mlx_lm.convert --quantize`), keeping everything in Apple's unified-memory format
  so there's no separate CPU/GPU copy.

## MLX vs. PyTorch

- MLX is Apple's array framework built specifically for Apple Silicon's **unified
  memory** architecture: CPU and GPU (and ANE) share the same physical memory, so
  MLX arrays can be operated on by either without copying.
- Lazy evaluation: MLX builds a computation graph and only materializes results
  when needed (similar to old TF1 graphs), which lets it fuse ops.
- No CUDA — this matters here because it means no `bitsandbytes`/`peft`/HF
  `transformers` full stack; `mlx-lm` is a separate, smaller ecosystem with its own
  LoRA (`mlx_lm.lora`), fusing (`mlx_lm.fuse`), and generation (`mlx_lm.generate`)
  CLIs purpose-built for this hardware.
- Practical implication for this project: model conversion/quantization must go
  through MLX's own tooling, not HF's `bitsandbytes`.

## Query rewriting techniques (for RAG)

- **Query expansion**: add synonyms/related terms to the original query to widen
  lexical/semantic recall.
- **Query decomposition** (this project's focus): break a multi-hop question into
  atomic sub-questions that can each be answered by a single retrieved passage,
  then merge results. Directly targets multi-hop QA where no single passage
  contains the full answer.
- **HyDE (Hypothetical Document Embeddings)**: generate a hypothetical *answer* to
  the query and embed that instead of the query itself — closes the
  question-vs-answer vocabulary/style mismatch that hurts dense retrieval
  (stretch goal, Phase 7.5).
- **Step-back prompting**: ask a more abstract/general version of the question
  first, to retrieve broader context, then answer the specific question with that
  context.

## Knowledge distillation (teacher → student)

- A large "teacher" model produces high-quality outputs (here: query
  decompositions) for a set of inputs; a smaller "student" model is trained to
  imitate them.
- Here it's **sequence-level / behavioral distillation via supervised
  fine-tuning**, not logit-matching distillation — the student never sees the
  teacher's probability distribution, only its final text output as a training
  label. This is standard practice for distilling API-only teacher models
  (Groq/Llama-3-70B) since token-level logits aren't exposed with equivalent
  tokenizers across model families anyway.
- Core assumption being tested (the project's "core hypothesis"): the student
  needs far fewer parameters than the teacher to reproduce this *narrow* skill
  (decomposition) even though it can't match the teacher's general capability.

## Retrieval metrics

- **Recall@k**: of the gold supporting passages for a question, what fraction are
  present in the top-k retrieved results. For multi-hop QA (HotpotQA has 2 gold
  supporting paragraphs per question), Recall@k is typically computed as
  "fraction of gold paragraphs found in top-k," averaged over the eval set. This
  is the primary metric in this project since it directly measures whether
  decomposition helped find the right evidence.
- **MRR (Mean Reciprocal Rank)**: average of `1/rank` of the first relevant result
  across queries. Rewards ranking relevant results higher, not just including them
  — useful as a secondary/quality signal alongside Recall@k.
- With decomposition, Recall@k is computed **per sub-question then merged**
  (union of retrieved passages across sub-questions, deduplicated) — this is the
  mechanism by which decomposition is expected to beat a single raw-query search
  for multi-hop questions.

## Catastrophic forgetting in fine-tuning

- Risk: fine-tuning on a narrow task (decomposition) can degrade the base model's
  general capabilities (fluency, instruction-following on unrelated tasks) if
  training runs too long, uses too high a learning rate, or the LoRA rank/target
  modules are too broad.
- LoRA is inherently more resistant than full fine-tuning since the base weights
  are frozen — but a large-magnitude adapter can still dominate the forward pass
  in target modules and effectively "overwrite" prior behavior for those
  computations.
- Mitigations used here: keep training data narrowly scoped to the single task,
  monitor validation loss and stop at the best checkpoint (Phase 5) rather than
  training to convergence on training loss, and qualitatively check the
  fine-tuned model still produces coherent, non-degenerate decompositions before
  calling it done (also feeds into the failure-case analysis in Phase 9).
