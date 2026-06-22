"""
Ingestion pipeline: PDF -> page-aware text extraction -> chunks -> embeddings -> ChromaDB.

Design decisions (worth knowing for interviews):
- We keep page numbers attached to every chunk. Without this, citations are
  impossible later, and "ground the answer in the document" becomes a lie.
- We use a recursive character splitter with overlap so we don't cut a
  sentence/clause in half at a chunk boundary, which would hurt retrieval
  quality on edge-of-chunk facts.
- Embeddings run locally via sentence-transformers (free, no API cost).
  Only the final answer-generation step calls out to a paid API (Groq).
"""
import os
import uuid
from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_file: str
    page_number: int


def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Returns list of (page_number, page_text), 1-indexed pages."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    return pages


def chunk_document(pdf_path: str) -> list[Chunk]:
    """Extract + split a single PDF into page-tagged chunks."""
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
    """Returns a persistent Chroma collection using a local sentence-transformers
    embedding function. This is the vectorizing step: every chunk of text gets
    converted into a 384-dim vector capturing its semantic meaning, so we can
    later search by "meaning similarity" rather than exact keyword match."""
    client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.EMBEDDING_MODEL
    )

    collection = client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for matching
    )
    return collection


def reset_collection():
    """Wipe the collection. Useful in dev when re-ingesting the same docs
    repeatedly, to avoid duplicate chunks piling up (see idempotency gap
    we discussed — this is the manual/dev-time workaround for now)."""
    client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
    try:
        client.delete_collection(settings.CHROMA_COLLECTION_NAME)
    except Exception:
        pass  # collection may not exist yet


def ingest_pdf(pdf_path: str) -> int:
    """Chunks a PDF, embeds each chunk, and writes it into Chroma.
    Returns the number of chunks ingested."""
    chunks = chunk_document(pdf_path)
    if not chunks:
        return 0

    collection = get_chroma_collection()

    # Chroma handles the embedding call internally via embed_fn when we .add()
    collection.add(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[
            {"source_file": c.source_file, "page_number": c.page_number} for c in chunks
        ],
    )
    return len(chunks)


def ingest_directory(dir_path: str) -> dict[str, int]:
    """Ingest every PDF in a directory. Returns {filename: chunk_count}."""
    results = {}
    for fname in sorted(os.listdir(dir_path)):
        if fname.lower().endswith(".pdf"):
            full_path = os.path.join(dir_path, fname)
            count = ingest_pdf(full_path)
            results[fname] = count
    return results


if __name__ == "__main__":
    # Manual smoke test: python -m app.ingestion [directory]
    import sys

    target_dir = sys.argv[1] if len(sys.argv) > 1 else "data/sample_docs"
    print(f"Resetting collection and ingesting PDFs from: {target_dir}")
    reset_collection()
    results = ingest_directory(target_dir)
    for fname, count in results.items():
        print(f"  {fname}: {count} chunks")
    print(f"Total files ingested: {len(results)}")