"""
Phase 7 — Self-contained RAG demo: decompose -> retrieve per sub-question
(FAISS over the HotpotQA context corpus) -> merge -> generate answer.

Usage:
    python src/rag_pipeline.py --model models/fused --question "..."
    python src/rag_pipeline.py --model models/fused --index 0     # a held-out test question
"""
import argparse
import json

import faiss
from sentence_transformers import SentenceTransformer

from decomposer import BASE_MODEL, LocalDecomposer
from eval_retrieval import EMBED_MODEL, build_shared_corpus, embed

ANSWER_PROMPT = (
    "Answer the question using only the passages below. Reply with a short answer.\n\n"
    "{passages}\n\nQuestion: {question}\nAnswer:"
)


class RagPipeline:
    def __init__(self, decomposer, answerer, corpus_file, per_query_k=3):
        with open(corpus_file) as f:
            records = [json.loads(l) for l in f if l.strip()]
        self.records = records
        self.paragraphs, self.titles = build_shared_corpus(records)
        self.embedder = SentenceTransformer(EMBED_MODEL)
        vecs = embed(self.embedder, self.paragraphs)
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        self.decomposer, self.answerer, self.k = decomposer, answerer, per_query_k

    def run(self, question: str) -> dict:
        sub_questions = self.decomposer.decompose(question)
        _, idxs = self.index.search(embed(self.embedder, sub_questions), self.k)
        picked = []
        for rank in range(self.k):
            for row in idxs:
                if row[rank] not in picked:
                    picked.append(int(row[rank]))
        passages = "\n\n".join(f"[{self.titles[i]}] {self.paragraphs[i]}" for i in picked)
        answer = self.answerer.generate(
            ANSWER_PROMPT.format(passages=passages, question=question), max_tokens=40
        ).split("<|eot_id|>")[0].strip()
        return {"question": question, "sub_questions": sub_questions,
                "retrieved_titles": [self.titles[i] for i in picked], "answer": answer}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE_MODEL, help="fused fine-tuned model dir (decomposition)")
    ap.add_argument("--corpus", default="data/processed/test_full.jsonl")
    ap.add_argument("--question")
    ap.add_argument("--index", type=int)
    args = ap.parse_args()

    decomposer = LocalDecomposer(args.model)
    answerer = LocalDecomposer(BASE_MODEL)
    rag = RagPipeline(decomposer, answerer, args.corpus)
    question = args.question or rag.records[args.index or 0]["question"]
    print(json.dumps(rag.run(question), indent=2))
    if args.question is None:
        print("gold answer:", rag.records[args.index or 0]["answer"])


if __name__ == "__main__":
    main()
