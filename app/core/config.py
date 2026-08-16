"""
Centralized configuration. Reads from environment variables (.env file).
Keeping config in one place is a production habit: nothing else in the
codebase should call os.getenv() directly.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --- LLM ---
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    # --- Embeddings ---
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # --- Vector store ---
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_store")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "hr_policy_docs")

    # --- Chunking ---
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 800))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 120))

    # --- Retrieval ---
    TOP_K_DENSE: int = int(os.getenv("TOP_K_DENSE", 5))
    TOP_K_BM25: int = int(os.getenv("TOP_K_BM25", 5))
    TOP_K_FINAL: int = int(os.getenv("TOP_K_FINAL", 4))

    # --- Cohere Rerank ---
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
    COHERE_RERANK_MODEL: str = os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0")
    TOP_K_RERANKED: int = int(os.getenv("TOP_K_RERANKED", 4))

    # --- LangSmith (observability/tracing) ---
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "docqa-hr-assistant")
    LANGSMITH_TRACING: bool = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

    def __init__(self):
        # LangChain/LangGraph tracing looks for these specific env var names.
        # We bridge our own .env naming to LangChain's expected names here,
        # in one place, so the rest of the app doesn't need to know about it.
        if self.LANGSMITH_TRACING:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.LANGSMITH_API_KEY
            os.environ["LANGCHAIN_PROJECT"] = self.LANGSMITH_PROJECT

    def validate(self):
        """Call this at startup so missing config fails fast and loud,
        not three layers deep inside a retrieval call."""
        missing = []
        if not self.GROQ_API_KEY:
            missing.append("GROQ_API_KEY")
        if not self.COHERE_API_KEY:
            missing.append("COHERE_API_KEY")
        if not self.LANGSMITH_API_KEY:
            missing.append("LANGSMITH_API_KEY")
        if missing:
            raise ValueError(
                f"Missing required env vars: {', '.join(missing)}. "
                f"Check your .env file against .env.example."
            )


settings = Settings()