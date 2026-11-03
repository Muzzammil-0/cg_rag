import sys
import time
import json
from pathlib import Path
from contextlib import asynccontextmanager, closing

from fastapi import FastAPI, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.api.models import QueryRequest, QueryResponse, RetrievedChunk, HealthResponse
from src.generation.generator import generate_answer
from src.storage.db import init_db, get_connection

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up...")
    init_db()
    print("DB initialized.")
    print("Startup complete.")
    yield
    print("Shutting down.")

app = FastAPI(
    title="CG Legal RAG",
    description="Chhattisgarh Judicial RAG — ask legal questions, get cited answers.",
    version="0.1.0",
    lifespan=lifespan,
)

def _log_query(question: str, answer_data: dict, results: list, latency_ms: int):
    """Persist one query to SQLite using a context manager for safety."""
    top = results[0] if results else None
    top_source = top.get("metadata", {}).get("source") if top else None
    top_section = None
    match_type = None
    if top:
        meta = top.get("metadata", {})
        top_section = str(meta.get("section", meta.get("article", ""))) or None
        match_type = top.get("match_type")

    try:
        with closing(get_connection()) as conn:
            conn.execute(
                """
                INSERT INTO queries
                  (question, answer, citations, confidence,
                   retrieved_top_source, retrieved_top_section, match_type, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    question,
                    answer_data.get("answer", ""),
                    json.dumps(answer_data.get("citations", [])),
                    float(answer_data.get("confidence", 0.0)),
                    top_source,
                    top_section,
                    match_type,
                    latency_ms,
                ),
            )
            conn.commit()
    except Exception as e:
        print(f"[log_query] failed during logging: {e}")

def _create_retrieved_chunks(results: list) -> list[RetrievedChunk]:
    """Helper function to transform raw search results into pydantic models."""
    retrieved = []
    for r in results:
        m = r.get("metadata", {})
        chunk_data = {
            "source": m.get("source", ""),
            "section": str(m.get("section", m.get("article", ""))) or "",
            "snippet": r.get("text", "")[:400],
            "match_type": r.get("match_type", ""),
        }
        retrieved.append(RetrievedChunk(**chunk_data))
    return retrieved

@app.get("/health", response_model=HealthResponse)
def health():
    """Basic liveness check endpoint."""
    return {"status": "ok"}

@app.get("/healthz")
def healthz():
    """Minimal status check endpoint."""
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    start = time.time()

    try:
        # Note: generate_answer in generator.py takes verbose=False by default in our V2 merge
        answer_data, results = generate_answer(req.question, top_k=req.top_k, verbose=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    latency_ms = int((time.time() - start) * 1000)

    _log_query(req.question, answer_data, results, latency_ms)
    retrieved = _create_retrieved_chunks(results)

    return QueryResponse(
        question=req.question,
        answer=answer_data.get("answer", ""),
        citations=answer_data.get("citations", []),
        confidence=float(answer_data.get("confidence", 0.0)),
        latency_ms=latency_ms,
        retrieved=retrieved,
    )