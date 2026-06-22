"""
Generation pipeline: retrieved chunks -> Groq LLM -> grounded answer with citations.

Orchestrated as a LangGraph graph with three nodes:
  retrieve -> check_confidence -> generate_answer (or decline)

Why a graph instead of one function: this structure is what lets us add
branching logic later (e.g. clarifying questions, fallback retrieval)
without rewriting the whole flow — each node has one clear job.
"""
import json
from typing import TypedDict

from groq import Groq
from langgraph.graph import StateGraph, END

from app.retrieval import hybrid_retrieve, RetrievedChunk
from app.config import settings

# Below this Cohere relevance score, we don't trust the retrieval enough
# to let the LLM answer — better to say "I don't know" than hallucinate.
CONFIDENCE_THRESHOLD = 0.3


class GraphState(TypedDict):
    query: str
    chunks: list[RetrievedChunk]
    confident: bool
    answer: str
    citations: list[dict]


def retrieve_node(state: GraphState) -> GraphState:
    chunks = hybrid_retrieve(state["query"])
    return {**state, "chunks": chunks}


def check_confidence_node(state: GraphState) -> GraphState:
    chunks = state["chunks"]
    confident = bool(chunks) and chunks[0].score >= CONFIDENCE_THRESHOLD
    return {**state, "confident": confident}


def decline_node(state: GraphState) -> GraphState:
    return {
        **state,
        "answer": (
            "I don't have enough information in the HR documents to answer "
            "that confidently. Could you rephrase the question, or check "
            "with HR directly?"
        ),
        "citations": [],
    }


def generate_answer_node(state: GraphState) -> GraphState:
    client = Groq(api_key=settings.GROQ_API_KEY)

    context_blocks = []
    for i, c in enumerate(state["chunks"], start=1):
        context_blocks.append(
            f"[Source {i}: {c.source_file}, page {c.page_number}]\n{c.text}"
        )
    context = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an HR policy assistant. Answer the user's question using "
        "ONLY the provided source excerpts. Do not use outside knowledge. "
        "Respond ONLY in valid JSON with this exact shape:\n"
        '{"answer": "<your answer>", '
        '"citations": [{"source_file": "<file>", "page_number": <int>}]}\n'
        "Cite every source you actually used. If the sources don't contain "
        "the answer, say so honestly in the answer field and return an "
        "empty citations list."
    )

    user_prompt = f"Sources:\n{context}\n\nQuestion: {state['query']}"

    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,  # low temperature: we want grounded, not creative
    )

    parsed = json.loads(response.choices[0].message.content)
    return {
        **state,
        "answer": parsed.get("answer", ""),
        "citations": parsed.get("citations", []),
    }


def route_after_confidence(state: GraphState) -> str:
    return "generate_answer" if state["confident"] else "decline"


def build_graph():
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
        {"generate_answer": "generate_answer", "decline": "decline"},
    )
    graph.add_edge("generate_answer", END)
    graph.add_edge("decline", END)

    return graph.compile()


def answer_question(query: str) -> dict:
    """Main entry point: runs the full graph, returns answer + citations."""
    app_graph = build_graph()
    result = app_graph.invoke({"query": query, "chunks": [], "confident": False, "answer": "", "citations": []})
    return {"answer": result["answer"], "citations": result["citations"]}


if __name__ == "__main__":
    # Manual smoke test: python -m app.generation "your question here"
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "How many vacation days do employees get?"
    print(f"Query: {query}\n")
    result = answer_question(query)
    print("Answer:")
    print(result["answer"])
    print("\nCitations:")
    for c in result["citations"]:
        print(f"  - {c['source_file']}, page {c['page_number']}")