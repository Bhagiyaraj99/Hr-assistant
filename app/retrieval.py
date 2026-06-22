"""
Hybrid retrieval: dense (embedding) search + BM25 (keyword) search,
merged with Reciprocal Rank Fusion (RRF).

Why hybrid: dense search is great at semantic similarity but can miss exact
terms (policy numbers, specific clause names). BM25 is great at exact
keyword matches but misses paraphrases. Combining both is a well-known
production pattern.
"""

import cohere
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.ingestion import get_chroma_collection
from app.config import settings


@dataclass
class RetrievedChunk:
    text: str
    source_file: str
    page_number: int
    score: float


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _load_corpus():
    """Pulls the full corpus from Chroma so we can build a BM25 index over it."""
    collection = get_chroma_collection()
    data = collection.get(include=["documents", "metadatas"])
    return data["ids"], data["documents"], data["metadatas"]


def dense_search(query: str, top_k: int) -> list[RetrievedChunk]:
    """Embedding-based search: finds chunks with similar MEANING to the query."""
    collection = get_chroma_collection()
    results = collection.query(query_texts=[query], n_results=top_k)

    chunks = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        similarity = 1 - dist  # convert distance -> similarity score
        chunks.append(
            RetrievedChunk(
                text=doc,
                source_file=meta["source_file"],
                page_number=meta["page_number"],
                score=similarity,
            )
        )
    return chunks


def bm25_search(query: str, top_k: int) -> list[RetrievedChunk]:
    """Keyword-based search: finds chunks with the most matching exact words."""
    ids, documents, metadatas = _load_corpus()
    if not documents:
        return []

    tokenized_corpus = [_tokenize(d) for d in documents]
    bm25 = BM25Okapi(tokenized_corpus)

    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(
        zip(documents, metadatas, scores), key=lambda x: x[2], reverse=True
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
    """Merge multiple ranked lists into one ranking using RRF.
    Score = sum over lists of 1/(k + rank). This avoids needing to compare
    two very different scoring scales (cosine similarity vs BM25 score) —
    it only cares about each chunk's RANK position in each list."""
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


def hybrid_retrieve(query: str) -> list[RetrievedChunk]:
    """Main entry point: dense + BM25 -> fuse -> rerank with Cohere."""
    dense_results = dense_search(query, settings.TOP_K_DENSE)
    bm25_results = bm25_search(query, settings.TOP_K_BM25)

    fused = reciprocal_rank_fusion([dense_results, bm25_results])
    reranked = rerank(query, fused)
    return reranked

def rerank(query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Re-score candidates using Cohere's rerank model, which reads the
    actual query + chunk text together and judges true relevance — far more
    precise than rank-position fusion (RRF) alone, since RRF never actually
    reads the content, just the rank order."""
    if not candidates:
        return []

    co = cohere.Client(settings.COHERE_API_KEY)

    response = co.rerank(
        model=settings.COHERE_RERANK_MODEL,
        query=query,
        documents=[c.text for c in candidates],
        top_n=settings.TOP_K_RERANKED,
    )

    reranked = []
    for result in response.results:
        original = candidates[result.index]
        reranked.append(
            RetrievedChunk(
                text=original.text,
                source_file=original.source_file,
                page_number=original.page_number,
                score=result.relevance_score,  # Cohere's own 0-1 relevance score
            )
        )
    return reranked

if __name__ == "__main__":
    # Manual smoke test: python -m app.retrieval "your question here"
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "How many vacation days do employees get?"
    print(f"Query: {query}\n")
    results = hybrid_retrieve(query)
    for i, r in enumerate(results, start=1):
        print(f"[{i}] score={r.score:.4f} | {r.source_file} p.{r.page_number}")
        print(f"    {r.text[:150]}...")
        print()