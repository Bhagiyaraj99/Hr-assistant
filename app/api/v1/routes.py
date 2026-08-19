"""
API v1 routes: /health, /query, /upload.

Routes are thin by design — they validate input, call a service,
and return a response. No business logic lives here.
"""

import os
import time

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.core.config import settings
from app.schemas.query import QueryRequest, QueryResponse, Citation, UploadResponse
from app.services.generation import answer_question
from app.services.ingestion import ingest_pdf

router = APIRouter()


@router.get("/health")
def health_check():
    """Liveness check — confirms the service is up and config is loaded."""
    return {
        "status": "ok",
        "llm_model": settings.LLM_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
    }


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Answer a question grounded in the ingested HR policy documents.
    Returns the answer, page-level citations, and end-to-end latency.
    """
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


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF and ingest it into the vector store.
    The document becomes immediately queryable via /query.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    upload_dir = "data/sample_docs"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    chunk_count = ingest_pdf(file_path)

    return UploadResponse(
        status="ingested",
        filename=file.filename,
        chunks_stored=chunk_count,
    )
