"""
Hybrid retrieval: dense (embedding) search + BM25 (keyword) search,
merged with Reciprocal Rank Fusion, then re-ranked by Cohere.

Why hybrid: dense search catches semantic similarity, BM25 catches exact
terms (policy numbers, clause names). RRF merges both without needing to
normalize their different scoring scales — it only cares about rank position.
Cohere rerank then does a precise final pass, reading query + chunk together.
"""

from dataclasses import dataclass

import cohere
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.services.ingestion import get_chroma_collection


@dataclass
class RetrievedChunk:
    """A chunk returned from retrieval, with its relevance score."""
    text: str
    source_file: str
    page_number: int
    score: float


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer for BM25."""
    return text.lower().split()


def _load_corpus() -> tuple:
    """Pull the full corpus from ChromaDB to build a BM25 index over it."""
    collection = get_chroma_collection()
    data = collection.get(include=["documents", "metadatas"])
    return data["ids"], data["documents"], data["metadatas"]


def dense_search(query: str, top_k: int) -> list[RetrievedChunk]:
    """Embedding-based search — finds chunks with similar meaning to the query."""
    collection = get_chroma_collection()
    results = collection.query(query_texts=[query], n_results=top_k)

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append(RetrievedChunk(
            text=doc,
            source_file=meta["source_file"],
            page_number=meta["page_number"],
            score=1 - dist,  # convert cosine distance → similarity
        ))
    return chunks


def bm25_search(query: str, top_k: int) -> list[RetrievedChunk]:
    """Keyword-based search — finds chunks with the most matching exact words."""
    _, documents, metadatas = _load_corpus()
    if not documents:
        return []

    tokenized_corpus = [_tokenize(d) for d in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(_tokenize(query))

    ranked = sorted(
        zip(documents, metadatas, scores),
        key=lambda x: x[2],
        reverse=True,
    )[:top_k]

    return [
        RetrievedChunk(
            text=doc,
            source_file=meta["source_file"],
            page_number=meta["page_number"],
            score=float(score),
        )
        for doc, meta, score in ranked
        if score > 0
    ]


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]], k: int = 60
) -> list[RetrievedChunk]:
    """
    Merge multiple ranked lists using RRF.

    Score = sum of 1/(k + rank) across all lists. Using rank position instead
    of raw scores avoids needing to normalize cosine similarity vs BM25 values.
    """
    fused_scores: dict[str, float] = {}
    chunk_lookup: dict[str, RetrievedChunk] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list):
            key = f"{chunk.source_file}::{chunk.page_number}::{chunk.text[:50]}"
            fused_scores[key] = fused_scores.get(key, 0) + 1 / (k + rank + 1)
            chunk_lookup[key] = chunk

    ranked_keys = sorted(fused_scores, key=lambda k_: fused_scores[k_], reverse=True)

    return [
        RetrievedChunk(
            text=chunk_lookup[key].text,
            source_file=chunk_lookup[key].source_file,
            page_number=chunk_lookup[key].page_number,
            score=fused_scores[key],
        )
        for key in ranked_keys
    ]


def rerank(query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """
    Re-score candidates using Cohere's rerank model.

    Unlike RRF (which never reads the content), Cohere reads the actual query
    and chunk text together, producing a genuine 0-1 relevance score. This
    score is meaningful — we use it downstream to decide whether to answer
    or decline (confidence threshold).
    """
    if not candidates:
        return []

    co = cohere.Client(settings.COHERE_API_KEY)
    response = co.rerank(
        model=settings.COHERE_RERANK_MODEL,
        query=query,
        documents=[c.text for c in candidates],
        top_n=settings.TOP_K_RERANKED,
    )

    return [
        RetrievedChunk(
            text=candidates[result.index].text,
            source_file=candidates[result.index].source_file,
            page_number=candidates[result.index].page_number,
            score=result.relevance_score,
        )
        for result in response.results
    ]


def hybrid_retrieve(query: str) -> list[RetrievedChunk]:
    """
    Main retrieval entry point.

    Flow: dense search + BM25 → RRF fusion → Cohere rerank.
    Returns the top reranked chunks with genuine relevance scores.
    """
    dense_results = dense_search(query, settings.TOP_K_DENSE)
    bm25_results = bm25_search(query, settings.TOP_K_BM25)
    fused = reciprocal_rank_fusion([dense_results, bm25_results])
    return rerank(query, fused)
