"""
Ingestion pipeline: PDF → chunks → embeddings → ChromaDB.

Stages: extract_pages → chunk_document → ingest_pdf → ingest_directory
Page numbers are attached to every chunk so citations work downstream.
"""

import os
import uuid
from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings


@dataclass
class Chunk:
    """A single text chunk with its origin tracked for citations."""
    chunk_id: str
    text: str
    source_file: str
    page_number: int  # 1-indexed


def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Extract text per page from a PDF. Skips blank/unreadable pages."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    return pages


def chunk_document(pdf_path: str) -> list[Chunk]:
    """
    Split a PDF into overlapping chunks, each tagged with page number.

    We chunk page-by-page (not the whole doc at once) so page metadata
    stays accurate. RecursiveCharacterTextSplitter tries paragraph → sentence
    → word breaks before hard cuts, keeping chunks semantically coherent.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    source_file = os.path.basename(pdf_path)
    chunks: list[Chunk] = []

    for page_number, page_text in extract_pages(pdf_path):
        for piece in splitter.split_text(page_text):
            if not piece.strip():
                continue
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    text=piece,
                    source_file=source_file,
                    page_number=page_number,
                )
            )
    return chunks


def get_chroma_collection():
    """
    Connect to the persistent ChromaDB collection.

    Embeddings run locally via sentence-transformers — no API cost.
    Cosine similarity is used as the distance metric, standard for semantic search.
    """
    client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.EMBEDDING_MODEL
    )
    return client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection() -> None:
    """
    Wipe the ChromaDB collection. Use during dev to avoid duplicate chunks
    when re-ingesting the same documents.
    """
    client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
    try:
        client.delete_collection(settings.CHROMA_COLLECTION_NAME)
    except Exception:
        pass


def ingest_pdf(pdf_path: str) -> int:
    """Chunk a PDF and store embeddings in ChromaDB. Returns chunk count."""
    chunks = chunk_document(pdf_path)
    if not chunks:
        return 0

    collection = get_chroma_collection()
    collection.add(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[
            {"source_file": c.source_file, "page_number": c.page_number}
            for c in chunks
        ],
    )
    return len(chunks)


def ingest_directory(dir_path: str) -> dict[str, int]:
    """Ingest all PDFs in a directory. Returns {filename: chunk_count}."""
    results = {}
    for fname in sorted(os.listdir(dir_path)):
        if fname.lower().endswith(".pdf"):
            full_path = os.path.join(dir_path, fname)
            results[fname] = ingest_pdf(full_path)
    return results
