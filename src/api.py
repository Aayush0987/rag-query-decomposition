"""
Phase 8 — FastAPI service exposing the fused fine-tuned decomposer.

    MODEL_PATH=models/fused uvicorn api:app --app-dir src --port 8000
    curl -X POST localhost:8000/decompose -H 'content-type: application/json' \
         -d '{"query": "What nationality is the director of Titanic?"}'
"""
import os
import time

from fastapi import FastAPI
from pydantic import BaseModel

from decomposer import BASE_MODEL, LocalDecomposer

app = FastAPI(title="Query Decomposition API")
decomposer = LocalDecomposer(os.environ.get("MODEL_PATH", BASE_MODEL))


class DecomposeRequest(BaseModel):
    query: str


@app.post("/decompose")
def decompose(req: DecomposeRequest):
    start = time.perf_counter()
    sub_questions = decomposer.decompose(req.query)
    return {"query": req.query, "sub_questions": sub_questions,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1)}


@app.get("/health")
def health():
    return {"status": "ok"}
