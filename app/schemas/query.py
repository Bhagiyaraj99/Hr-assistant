"""
Pydantic schemas for the /query and /upload endpoints.

Keeping schemas separate from routes means they can be reused across
multiple endpoints and are easy to version when the API evolves.
"""

from pydantic import BaseModel


class QueryRequest(BaseModel):
    """Incoming question from the user."""
    question: str

    model_config = {
        "json_schema_extra": {
            "example": {"question": "How many vacation days do employees get?"}
        }
    }


class Citation(BaseModel):
    """A single source reference returned with an answer."""
    source_file: str
    page_number: int


class QueryResponse(BaseModel):
    """Full response returned by the /query endpoint."""
    answer: str
    citations: list[Citation]
    latency_ms: float


class UploadResponse(BaseModel):
    """Response returned after a successful PDF upload and ingestion."""
    status: str
    filename: str
    chunks_stored: int

