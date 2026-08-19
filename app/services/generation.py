"""
Generation pipeline: retrieved chunks → Groq LLM → grounded answer with citations.

Orchestrated as a LangGraph graph with three nodes:
    retrieve → check_confidence → generate_answer (or decline)

Confidence gating: if the top Cohere rerank score is below the threshold,
we decline gracefully instead of letting the LLM guess — this is the core
hallucination-prevention mechanism.
"""

import json
from typing import TypedDict

from groq import Groq
from langgraph.graph import StateGraph, END

from app.core.config import settings
from app.services.retrieval import hybrid_retrieve, RetrievedChunk

# If top Cohere score is below this, we don't trust retrieval enough to answer.
CONFIDENCE_THRESHOLD = 0.3


# ─────────────────────────────────────────────
# Graph state
# ─────────────────────────────────────────────

class GraphState(TypedDict):
    """Shared state passed between every node in the LangGraph."""
    query: str
    chunks: list[RetrievedChunk]
    confident: bool
    answer: str
    citations: list[dict]


# ─────────────────────────────────────────────
# Graph nodes
# ─────────────────────────────────────────────

def retrieve_node(state: GraphState) -> GraphState:
    """Run hybrid retrieval and store results in state."""
    chunks = hybrid_retrieve(state["query"])
    return {**state, "chunks": chunks}


def check_confidence_node(state: GraphState) -> GraphState:
    """
    Check if the top retrieved chunk meets the confidence threshold.
    Sets state['confident'] which controls the routing decision next.
    """
    chunks = state["chunks"]
    confident = bool(chunks) and chunks[0].score >= CONFIDENCE_THRESHOLD
    return {**state, "confident": confident}


def generate_answer_node(state: GraphState) -> GraphState:
    """
    Send retrieved chunks + query to Groq and get a grounded JSON answer.

    We force JSON output with response_format so citations come back
    as structured data, not free text we'd have to parse unreliably.
    Temperature is set low (0.1) — we want factual, not creative.
    """
    client = Groq(api_key=settings.GROQ_API_KEY)

    # Build context block with source labels so the LLM can cite correctly
    context_blocks = []
    for i, chunk in enumerate(state["chunks"], start=1):
        context_blocks.append(
            f"[Source {i}: {chunk.source_file}, page {chunk.page_number}]\n{chunk.text}"
        )
    context = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an HR policy assistant. Answer using ONLY the provided sources. "
        "Do not use outside knowledge. "
        "Respond ONLY in valid JSON with this exact shape:\n"
        '{"answer": "<your answer>", '
        '"citations": [{"source_file": "<file>", "page_number": <int>}]}\n'
        "Cite every source you used. If sources don't contain the answer, "
        "say so in the answer field and return an empty citations list."
    )

    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {state['query']}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    parsed = json.loads(response.choices[0].message.content)
    return {
        **state,
        "answer": parsed.get("answer", ""),
        "citations": parsed.get("citations", []),
    }


def decline_node(state: GraphState) -> GraphState:
    """Return a honest 'I don't know' response when confidence is too low."""
    return {
        **state,
        "answer": (
            "I don't have enough information in the HR documents to answer "
            "that confidently. Please rephrase your question or check with HR directly."
        ),
        "citations": [],
    }


# ─────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────

def route_after_confidence(state: GraphState) -> str:
    """Route to generation if confident, decline otherwise."""
    return "generate_answer" if state["confident"] else "decline"


# ─────────────────────────────────────────────
# Graph assembly
# ─────────────────────────────────────────────

def build_graph():
    """
    Assemble and compile the LangGraph pipeline.

    Graph structure:
        retrieve → check_confidence → [generate_answer | decline] → END
    """
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("check_confidence", check_confidence_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("decline", decline_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "check_confidence")
    graph.add_conditional_edges(
        "check_confidence",
        route_after_confidence,
        {
            "generate_answer": "generate_answer",
            "decline": "decline",
        },
    )
    graph.add_edge("generate_answer", END)
    graph.add_edge("decline", END)

    return graph.compile()


# ─────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────

def answer_question(query: str) -> dict:
    """
    Main entry point for the generation pipeline.

    Runs the full LangGraph: retrieve → confidence check → generate/decline.
    Returns a dict with 'answer' and 'citations' keys.
    """
    pipeline = build_graph()
    result = pipeline.invoke({
        "query": query,
        "chunks": [],
        "confident": False,
        "answer": "",
        "citations": [],
    })
    return {
        "answer": result["answer"],
        "citations": result["citations"],
    }
