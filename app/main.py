"""
Application entrypoint. Creates the FastAPI app and registers routes.

Run locally: uvicorn app.main:app --reload
"""

from fastapi import FastAPI

from app.api.v1.routes import router

app = FastAPI(
    title="DocQA — HR Policy & Handbook Assistant",
    description="RAG API that answers HR policy questions with page-level citations.",
    version="0.1.0",
)

app.include_router(router)
