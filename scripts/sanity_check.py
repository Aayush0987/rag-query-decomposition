"""
Phase 1 — Sanity check: confirm the quantized base model loads and generates
coherent text locally via MLX.

Usage:
    python scripts/sanity_check.py
"""

from mlx_lm import load, generate

MODEL = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"


def main():
    print(f"Loading {MODEL}...")
    model, tokenizer = load(MODEL)

    prompt = "Decompose this question into 2 atomic sub-questions: What nationality is the director of Titanic?"
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    print("Generating...")
    response = generate(model, tokenizer, prompt=formatted, max_tokens=150, verbose=True)
    print("\n--- Response ---")
    print(response)


if __name__ == "__main__":
    main()
