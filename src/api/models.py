from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(5, ge=1, le=20)


class RetrievedChunk(BaseModel):
    source: str
    section: str
    snippet: str
    match_type: str = ""


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list
    confidence: float
    latency_ms: int
    retrieved: list[RetrievedChunk]


class HealthResponse(BaseModel):
    status: str