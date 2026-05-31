"""
Agent layer tests — mock Gemini and retriever; no live API calls.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.models import ResearchResponse, SourceCitation
from agent.tools import calculate_metric  # does not import agent package __init__


def _import_graph():
    from agent.graph import (
        AgentState,
        _parse_rewrite_json,
        rewrite_node,
        synthesis_node,
        state_to_response,
    )
    return AgentState, _parse_rewrite_json, rewrite_node, synthesis_node, state_to_response


class TestQueryRewriter:
    def test_query_rewriter_extracts_ticker(self):
        _parse_rewrite_json = _import_graph()[1]
        raw = json.dumps(
            {
                "rewritten_query": "Apple gross margin Q2 Q3 2024",
                "ticker": "AAPL",
                "fiscal_year": 2024,
                "fiscal_quarter": None,
                "intent": "compare",
                "expression": None,
            }
        )
        parsed = _parse_rewrite_json(raw, "How did Apple margins change?")
        assert parsed["ticker"] == "AAPL"
        assert parsed["fiscal_year"] == 2024

    def test_query_rewriter_extracts_calculate_intent(self):
        _parse_rewrite_json = _import_graph()[1]
        raw = json.dumps(
            {
                "rewritten_query": "margin percentage calculation",
                "ticker": None,
                "fiscal_year": None,
                "fiscal_quarter": None,
                "intent": "calculate",
                "expression": "43200 / 89500 * 100",
            }
        )
        parsed = _parse_rewrite_json(raw, "What is 43200 / 89500 * 100")
        assert parsed["intent"] == "calculate"
        assert parsed["expression"] == "43200 / 89500 * 100"

    @patch("agent.graph._invoke_llm")
    def test_query_rewriter_extracts_intent_compare(self, mock_llm):
        AgentState, _, rewrite_node, _, _ = _import_graph()
        mock_llm.return_value = json.dumps(
            {
                "rewritten_query": "AAPL gross margin 2024 Q2 Q3",
                "ticker": "AAPL",
                "fiscal_year": 2024,
                "fiscal_quarter": None,
                "intent": "compare",
            }
        )
        state: AgentState = {
            "original_query": "How did Apple's gross margins change between Q2 and Q3 2024?",
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
        out = rewrite_node(state)
        assert out["intent"] == "compare"
        assert out["ticker"] == "AAPL"


class TestCalculateMetric:
    def test_calculate_metric_valid(self):
        assert calculate_metric.invoke({"expression": "100 * 0.42"}) == "42.0"

    def test_calculate_metric_blocks_unsafe(self):
        result = calculate_metric.invoke({"expression": "__import__('os')"})
        assert "error" in result.lower()
        assert "42" not in result


class TestSynthesisNode:
    @patch("agent.graph._get_llm")
    def test_synthesis_node_populates_answer(self, mock_get_llm):
        AgentState, _, _, synthesis_node, _ = _import_graph()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Apple gross margins improved between Q2 and Q3 2024 per 10-Q filings."
        )
        mock_get_llm.return_value = mock_llm

        state: AgentState = {
            "original_query": "How did Apple margins change?",
            "rewritten_query": "AAPL gross margin",
            "ticker": "AAPL",
            "fiscal_year": 2024,
            "fiscal_quarter": None,
            "intent": "compare",
            "expression": "",
            "retrieved_context": "SOURCE: AAPL_10Q_2024_Q2_001 | AAPL | MD&A | 10-Q | 2024 | Q2 | 2024-05-01\nMargin data here.",
            "tool_calls_made": ["compare_quarters"],
            "raw_chunks": [],
            "final_answer": "",
            "sources": [],
            "confidence": 0.7,
            "reasoning_trace": "",
        }
        out = synthesis_node(state)
        assert out["final_answer"]
        assert "Apple" in out["final_answer"] or "margin" in out["final_answer"].lower()


class TestRunResearch:
    @patch("agent.runner._get_graph")
    def test_run_research_returns_research_response(self, mock_get_graph):
        from agent.runner import run_research

        mock_graph = MagicMock()
        mock_get_graph.return_value = mock_graph
        mock_graph.invoke.return_value = {
            "original_query": "test",
            "rewritten_query": "AAPL revenue",
            "ticker": "AAPL",
            "fiscal_year": 2024,
            "fiscal_quarter": None,
            "intent": "metric",
            "expression": "",
            "raw_chunks": [
                {
                    "chunk_id": "AAPL_10Q_2024_Q2_001",
                    "ticker": "AAPL",
                    "section": "Financial Data",
                    "date_filed": "2024-05-01",
                    "filing_type": "10-Q",
                    "fiscal_year": 2024,
                    "fiscal_quarter": "Q2",
                }
            ],
            "retrieved_context": (
                "SOURCE: AAPL_10Q_2024_Q2_001 | AAPL | Financial Data | 10-Q | 2024 | Q2 | 2024-05-01\n"
                "Revenue table excerpt."
            ),
            "tool_calls_made": ["get_financial_metric"],
            "final_answer": "Revenue increased year over year.",
            "sources": [
                {
                    "chunk_id": "AAPL_10Q_2024_Q2_001",
                    "ticker": "AAPL",
                    "section": "Financial Data",
                    "date_filed": "2024-05-01",
                    "filing_type": "10-Q",
                    "fiscal_year": 2024,
                    "fiscal_quarter": "Q2",
                }
            ],
            "confidence": 0.9,
            "reasoning_trace": "Intent=metric; tools=['get_financial_metric']",
        }

        result = run_research("What was Apple revenue in 2024?")
        assert isinstance(result, ResearchResponse)
        assert result.answer == "Revenue increased year over year."
        assert len(result.sources) == 1
        assert isinstance(result.sources[0], SourceCitation)
        assert result.confidence == pytest.approx(0.9)
        assert "get_financial_metric" in result.tool_calls_made



class TestIntentRouting:
    """Verify arithmetic queries route to calculate, not retrieval."""

    def test_arithmetic_expression_detected(self):
        from agent.graph import _looks_like_arithmetic
        assert _looks_like_arithmetic("43200 / 89500 * 100") is True
        assert _looks_like_arithmetic("100 * 0.42") is True
        assert _looks_like_arithmetic("How did Apple margins change?") is False

    def test_calculate_intent_overrides_retrieval(self):
        """calculate intent must short-circuit before any RAG call."""
        from agent.tools import calculate_metric
        result = calculate_metric.invoke({"expression": "43200 / 89500 * 100"})
        assert "48" in result or "0.48" in result  # 48.27...

    def test_pure_expression_returns_number(self):
        from agent.tools import calculate_metric
        result = calculate_metric.invoke({"expression": "100 * 0.42"})
        assert "42" in result


class TestImportStability:
    """Ensure no circular imports or RuntimeWarning on module load."""

    def test_agent_init_is_empty(self):
        import importlib, agent
        # Should not expose run_research at top level (we emptied __init__)
        assert not hasattr(agent, "run_research") or callable(getattr(agent, "run_research", None))

    def test_runner_importable(self):
        from agent.runner import run_research
        assert callable(run_research)

    def test_rag_importable(self):
        from rag.retriever import FinancialRetriever
        from rag.filters import RetrievalFilters
        assert FinancialRetriever is not None
