"""
Entry point for the financial research agent.
"""

from __future__ import annotations

import json
import sys

from loguru import logger

from agent.graph import build_graph, state_to_response
from agent.models import ResearchResponse

_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_research(query: str) -> ResearchResponse:
    """Compile and run the research graph; always returns ResearchResponse."""
    initial = {
        "original_query": query,
        "rewritten_query": "",
        "ticker": None,
        "fiscal_year": None,
        "fiscal_quarter": None,
        "intent": "general",
        "expression": "",
        "retrieved_context": "",
        "tool_calls_made": [],
        "raw_chunks": [],
        "final_answer": "",
        "sources": [],
        "confidence": 0.4,
        "reasoning_trace": "",
    }
    try:
        final_state = _get_graph().invoke(initial)
        return state_to_response(final_state)
    except Exception as exc:
        logger.error(f"run_research failed: {exc}")
        return ResearchResponse(
            answer=f"Research failed: {exc}",
            sources=[],
            confidence=0.0,
            tool_calls_made=[],
            reasoning_trace=f"Graph error: {exc}",
        )


def main() -> None:
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Research query: ").strip()
    if not query:
        print(json.dumps({"error": "No query provided"}, indent=2))
        sys.exit(1)

    result = run_research(query)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
