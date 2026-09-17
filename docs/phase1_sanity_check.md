# Phase 1 — Environment Setup: Sanity Check

- Model: `mlx-community/Meta-Llama-3-8B-Instruct-4bit` (pre-quantized, downloaded via `huggingface_hub`)
- Framework: `mlx` 0.32.2, `mlx-lm`
- Hardware: Apple M4 Pro, 24GB unified memory

## Run

```
python scripts/sanity_check.py
```

## Result

Model loaded and generated a correct decomposition for the test prompt
("Decompose this question into 2 atomic sub-questions: What nationality is the
director of Titanic?"):

```
1. Who is the director of Titanic?
2. What nationality is [the director of Titanic]?
```

Generation continued past the intended stop point into a rambling
self-dialogue (the model didn't reliably stop at `<|eot_id|>`), which is a
known base-model quirk — will be handled in the real pipeline by trimming to
the first stop token / small `max_tokens` and by only using the first N
non-empty lines.

Performance: 33 tok/s prompt processing, 56 tok/s generation, ~5.4GB peak
memory — comfortably within the 24GB budget, leaving plenty of headroom for
LoRA fine-tuning in Phase 5.

**Conclusion: environment is set up correctly, model loads and generates
coherent, on-topic text locally. Phase 1 complete.**
