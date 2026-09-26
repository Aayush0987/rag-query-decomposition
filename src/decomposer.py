"""Local MLX decomposer used by the API, RAG pipeline, and benchmark."""
import threading

from mlx_lm import generate, load

from prompts import build_prompt, parse_subquestions

BASE_MODEL = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"


class LocalDecomposer:
    def __init__(self, model_path: str = BASE_MODEL, adapter_path: str | None = None):
        self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
        self._lock = threading.Lock()  # MLX generation is not thread-safe

    def generate(self, prompt: str, max_tokens: int = 150) -> str:
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        with self._lock:
            return generate(self.model, self.tokenizer, prompt=formatted,
                            max_tokens=max_tokens, verbose=False)

    def decompose(self, question: str) -> list[str]:
        return parse_subquestions(self.generate(build_prompt(question))) or [question]
