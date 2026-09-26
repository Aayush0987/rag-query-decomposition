"""Shared decomposition prompt so training, evaluation, and serving all use the same format."""

DECOMPOSE_INSTRUCTION = (
    "Decompose this multi-hop question into atomic sub-questions, one per line, "
    "numbered. Output only the sub-questions.\n\nQuestion: {question}\nSub-questions:"
)


def build_prompt(question: str) -> str:
    return DECOMPOSE_INSTRUCTION.format(question=question)


def parse_subquestions(text: str) -> list[str]:
    text = text.split("<|eot_id|>")[0]
    subqs = []
    for line in text.splitlines():
        cleaned = line.strip().lstrip("0123456789.-) ").strip()
        if cleaned:
            subqs.append(cleaned)
    return subqs


def format_completion(sub_questions: list[str]) -> str:
    return "\n".join(f"{i}. {q}" for i, q in enumerate(sub_questions, 1))
