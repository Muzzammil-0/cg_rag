import sys
from pathlib import Path

# Ensure repo root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.generation.generator import generate_answer

app = FastAPI(title="CG Legal RAG API", version="0.1")

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    citations: list
    confidence: float
    retrieved_contexts: list

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        answer_data, retrieved = generate_answer(request.question, top_k=request.top_k)
        # Prepare retrieved context summaries
        contexts = [res["text"][:300] for res in retrieved]
        return QueryResponse(
            answer=answer_data.get("answer", ""),
            citations=answer_data.get("citations", []),
            confidence=answer_data.get("confidence", 0.0),
            retrieved_contexts=contexts
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}