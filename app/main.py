"""
FastAPI application entrypoint. Exposes the RAG pipeline as a web service.

Run locally with: uvicorn app.main:app --reload
"""
import os
import time

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from app.config import settings
from app.generation import answer_question
from app.ingestion import ingest_pdf

app = FastAPI(
    title="DocQA — HR Policy & Handbook Assistant",
    description="RAG API that answers HR policy questions with page-level citations.",
    version="0.1.0",
)


# --- Request / Response models ---

class QueryRequest(BaseModel):
    question: str

    class Config:
        json_schema_extra = {
            "example": {"question": "How many vacation days do employees get?"}
        }


class Citation(BaseModel):
    source_file: str
    page_number: int


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    latency_ms: float


# --- Endpoints ---

@app.get("/health")
def health_check():
    """Basic liveness check — confirms the service is up and config loaded.
    This is what a cloud platform/load balancer pings to know if the
    container is healthy and ready to receive traffic."""
    return {
        "status": "ok",
        "llm_model": settings.LLM_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Answer a question grounded in the ingested HR policy documents.
    Returns the answer, page-level citations, and end-to-end latency."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    start = time.time()
    result = answer_question(request.question)
    latency_ms = round((time.time() - start) * 1000, 2)

    return QueryResponse(
        answer=result["answer"],
        citations=[Citation(**c) for c in result["citations"]],
        latency_ms=latency_ms,
    )


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF and ingest it into the vector store.
    Accepts any PDF file, chunks it, embeds it, and stores it in ChromaDB
    so it becomes immediately queryable via /query."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    upload_dir = "data/sample_docs"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    chunk_count = ingest_pdf(file_path)

    return {
        "status": "ingested",
        "filename": file.filename,
        "chunks_stored": chunk_count,
    }