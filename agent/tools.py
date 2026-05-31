"""
LangGraph tools for financial research.
HybridRetriever wraps Phase 2 FinancialRetriever with comparison and table preference helpers.
"""

from __future__ import annotations

import ast
import operator
import re
from functools import lru_cache
from typing import Any, Optional

from langchain_core.tools import tool
from loguru import logger

from rag.filters import RetrievalFilters

CHUNK_SEPARATOR = "\n\n---\n\n"
SOURCE_HEADER_RE = re.compile(
    r"^SOURCE:\s*(?P<chunk_id>\S+)\s*\|\s*(?P<ticker>\S+)\s*\|\s*"
    r"(?P<section>[^|]+)\s*\|\s*(?P<filing_type>\S+)\s*\|\s*"
    r"(?P<fiscal_year>[^|]*)\s*\|\s*(?P<fiscal_quarter>[^|]*)\s*\|\s*(?P<date_filed>\S+)",
    re.MULTILINE,
)

_SAFE_FUNCTIONS = {"abs": abs, "round": round, "min": min, "max": max}
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class HybridRetriever:
    """Agent-facing retriever with comparison and table-preference helpers."""

    def __init__(self, **kwargs) -> None:
        from rag.retriever import FinancialRetriever

        self._retriever = FinancialRetriever(**kwargs)

    def retrieve(
        self,
        query: str,
        filters: Optional[RetrievalFilters] = None,
        top_k: int = 8,
        alpha: float | None = None,
        prefer_tables: bool = False,
    ):
        from config import HYBRID_ALPHA, TOP_K

        kwargs: dict[str, Any] = {
            "query": query,
            "filters": filters,
            "top_k": top_k if top_k != 8 else TOP_K,
            "alpha": alpha if alpha is not None else HYBRID_ALPHA,
        }
        results = self._retriever.retrieve(**kwargs)
        if prefer_tables:
            tables = [c for c in results if c.metadata.get("chunk_type") == "table"]
            texts = [c for c in results if c.metadata.get("chunk_type") != "table"]
            results = (tables + texts)[: kwargs["top_k"]]
        return results

    def retrieve_for_comparison(
        self,
        ticker: str,
        metric: str,
        period1: str,
        period2: str,
        top_k: int = 5,
    ) -> tuple[list[Any], list[Any]]:
        """Retrieve chunks for two fiscal periods side by side."""
        query = metric
        chunks1 = self._retrieve_period(ticker, query, period1, top_k)
        chunks2 = self._retrieve_period(ticker, query, period2, top_k)
        return chunks1, chunks2

    def _retrieve_period(self, ticker: str, query: str, period: str, top_k: int):
        year, quarter = _parse_period(period)
        filters = RetrievalFilters(
            ticker=ticker.upper(),
            fiscal_year=year,
            fiscal_quarter=quarter,
        )
        try:
            return self.retrieve(query, filters=filters, top_k=top_k)
        except Exception as exc:
            logger.warning(f"Retrieval failed for {ticker} {period}: {exc}")
            return []


def _parse_period(period: str) -> tuple[int, Optional[str]]:
    """Parse '2024_Q2' into (2024, 'Q2')."""
    match = re.match(r"^(\d{4})_?(Q[1-4])?$", period.strip(), re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid period format: {period}. Expected e.g. 2024_Q2")
    year = int(match.group(1))
    quarter = match.group(2).upper() if match.group(2) else None
    return year, quarter


def _format_chunk(chunk) -> str:
    meta = chunk.metadata
    header = (
        f"SOURCE: {chunk.chunk_id} | {meta.get('ticker', '')} | "
        f"{meta.get('section', '')} | {meta.get('filing_type', '')} | "
        f"{meta.get('fiscal_year', '')} | {meta.get('fiscal_quarter') or 'annual'} | "
        f"{meta.get('date_filed', '')}"
    )
    return f"{header}\n{chunk.text}"


def _format_chunks(chunks: list, label: str) -> str:
    if not chunks:
        return f"### {label}\n_No chunks found for this period._"
    body = CHUNK_SEPARATOR.join(_format_chunk(c) for c in chunks)
    return f"### {label}\n{body}"


def chunks_to_raw_dicts(chunks: list) -> list[dict[str, Any]]:
    """Convert RetrievedChunk objects to citation metadata dicts."""
    out: list[dict[str, Any]] = []
    for c in chunks:
        meta = c.metadata or {}
        out.append(
            {
                "chunk_id": c.chunk_id or meta.get("chunk_id", ""),
                "ticker": meta.get("ticker", ""),
                "section": meta.get("section", ""),
                "date_filed": meta.get("date_filed", ""),
                "filing_type": meta.get("filing_type", ""),
                "fiscal_year": meta.get("fiscal_year"),
                "fiscal_quarter": meta.get("fiscal_quarter"),
            }
        )
    return out


def run_compare_quarters(
    ticker: str, metric: str, period1: str, period2: str
) -> tuple[str, list[dict[str, Any]]]:
    retriever = _get_retriever()
    try:
        chunks1, chunks2 = retriever.retrieve_for_comparison(
            ticker, metric, period1, period2
        )
    except Exception as exc:
        logger.error(f"compare_quarters failed: {exc}")
        return f"Comparison error: {exc}", []

    p1_label = f"{ticker.upper()} — {period1} ({metric})"
    p2_label = f"{ticker.upper()} — {period2} ({metric})"
    text = (
        f"## Quarter comparison: {metric}\n\n"
        f"{_format_chunks(chunks1, p1_label)}\n\n"
        f"{_format_chunks(chunks2, p2_label)}"
    )
    return text, chunks_to_raw_dicts(chunks1 + chunks2)


def run_get_risk_factors(ticker: str, fiscal_year: int) -> tuple[str, list[dict[str, Any]]]:
    retriever = _get_retriever()
    filters = RetrievalFilters(
        ticker=ticker.upper(),
        fiscal_year=fiscal_year,
        section="Risk Factors",
    )
    try:
        chunks = retriever.retrieve(f"{ticker} risk factors", filters=filters, top_k=5)
    except Exception as exc:
        logger.error(f"get_risk_factors failed: {exc}")
        return f"No relevant filings found. ({exc})", []

    if not chunks:
        return "No relevant Risk Factors chunks found for this ticker and year.", []
    chunks = chunks[:5]
    return CHUNK_SEPARATOR.join(_format_chunk(c) for c in chunks), chunks_to_raw_dicts(chunks)


def run_get_financial_metric(
    ticker: str,
    metric_query: str,
    fiscal_year: int,
    fiscal_quarter: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    retriever = _get_retriever()
    filters = RetrievalFilters(
        ticker=ticker.upper(),
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
    )
    try:
        chunks = retriever.retrieve(
            metric_query,
            filters=filters,
            top_k=5,
            prefer_tables=True,
        )
    except Exception as exc:
        logger.error(f"get_financial_metric failed: {exc}")
        return f"No relevant filings found. ({exc})", []

    if not chunks:
        return "No relevant metric chunks found.", []
    chunks = chunks[:5]
    return CHUNK_SEPARATOR.join(_format_chunk(c) for c in chunks), chunks_to_raw_dicts(chunks)


def run_generate_investment_thesis(ticker: str) -> tuple[str, list[dict[str, Any]]]:
    retriever = _get_retriever()
    try:
        mda_chunks = retriever.retrieve(
            "management discussion analysis business outlook",
            filters=RetrievalFilters(ticker=ticker.upper()),
            top_k=8,
        )
        mda_top = [
            c
            for c in mda_chunks
            if "md&a" in c.metadata.get("section", "").lower()
            or "management" in c.metadata.get("section", "").lower()
        ][:4]
        if not mda_top:
            mda_top = mda_chunks[:4]

        risk_chunks = retriever.retrieve(
            "risk factors",
            filters=RetrievalFilters(ticker=ticker.upper(), section="Risk Factors"),
            top_k=8,
        )
        risk_top = risk_chunks[:4]
    except Exception as exc:
        logger.error(f"generate_investment_thesis failed: {exc}")
        return f"No relevant filings found. ({exc})", []

    mda_text = (
        CHUNK_SEPARATOR.join(_format_chunk(c) for c in mda_top)
        if mda_top
        else "_No MD&A chunks found._"
    )
    risk_text = (
        CHUNK_SEPARATOR.join(_format_chunk(c) for c in risk_top)
        if risk_top
        else "_No Risk Factors chunks found._"
    )
    text = f"[MD&A]\n{mda_text}\n\n[RISK FACTORS]\n{risk_text}"
    return text, chunks_to_raw_dicts(mda_top + risk_top)


@lru_cache(maxsize=1)
def _get_retriever() -> HybridRetriever:
    return HybridRetriever()


def parse_sources_from_context(context: str) -> list[dict[str, Any]]:
    """Extract source citation dicts from formatted tool output."""
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for match in SOURCE_HEADER_RE.finditer(context):
        chunk_id = match.group("chunk_id")
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        fy = match.group("fiscal_year").strip()
        fq = match.group("fiscal_quarter").strip()
        sources.append(
            {
                "chunk_id": chunk_id,
                "ticker": match.group("ticker"),
                "section": match.group("section").strip(),
                "date_filed": match.group("date_filed"),
                "filing_type": match.group("filing_type"),
                "fiscal_year": int(fy) if fy.isdigit() else None,
                "fiscal_quarter": None if fq.lower() == "annual" else fq or None,
            }
        )
    return sources


def _safe_eval_expression(expression: str) -> str:
    """Evaluate a numeric expression with restricted operators and builtins."""
    expr = expression.strip()
    if not expr:
        return "Calculation error: empty expression"

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        return f"Calculation error: {exc}"

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
            return _SAFE_OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
            return _SAFE_OPERATORS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func = node.func.id
            if func not in _SAFE_FUNCTIONS:
                raise ValueError(f"Function not allowed: {func}")
            args = [_eval(a) for a in node.args]
            return _SAFE_FUNCTIONS[func](*args)
        raise ValueError(f"Unsupported expression element: {ast.dump(node)}")

    try:
        result = _eval(tree)
        return str(float(result) if isinstance(result, (int, float)) else result)
    except Exception as exc:
        return f"Calculation error: {exc}"


@tool
def compare_quarters(
    ticker: str, metric: str, period1: str, period2: str
) -> str:
    """Compare a financial metric between two fiscal quarters using SEC filing chunks.

    Args:
        ticker: Stock ticker (e.g. AAPL).
        metric: Metric to compare (e.g. gross margin, revenue).
        period1: First period as YYYY_QN (e.g. 2024_Q2).
        period2: Second period as YYYY_QN (e.g. 2024_Q3).
    """
    text, _ = run_compare_quarters(ticker, metric, period1, period2)
    return text


@tool
def get_risk_factors(ticker: str, fiscal_year: int) -> str:
    """Retrieve Risk Factors section excerpts from SEC filings for a ticker and year.

    Args:
        ticker: Stock ticker (e.g. MSFT).
        fiscal_year: 4-digit fiscal year (e.g. 2024).
    """
    text, _ = run_get_risk_factors(ticker, fiscal_year)
    return text


@tool
def get_financial_metric(
    ticker: str,
    metric_query: str,
    fiscal_year: int,
    fiscal_quarter: str | None = None,
) -> str:
    """Retrieve financial metric data from SEC filings, preferring table chunks.

    Args:
        ticker: Stock ticker.
        metric_query: Natural language metric query (e.g. gross margin revenue).
        fiscal_year: 4-digit fiscal year.
        fiscal_quarter: Optional quarter Q1-Q4; None for annual focus.
    """
    text, _ = run_get_financial_metric(ticker, metric_query, fiscal_year, fiscal_quarter)
    return text


@tool
def calculate_metric(expression: str) -> str:
    """Safely evaluate a numeric expression for financial calculations.

    Allowed: digits, + - * / ( ) . %, and functions abs, round, min, max.

    Args:
        expression: Arithmetic expression (e.g. '100 * 0.42').
    """
    if "__" in expression or "import" in expression.lower():
        return "Calculation error: unsafe expression"
    return _safe_eval_expression(expression)


@tool
def generate_investment_thesis(ticker: str) -> str:
    """Gather MD&A and Risk Factors excerpts to support an investment thesis.

    Args:
        ticker: Stock ticker to research.
    """
    text, _ = run_generate_investment_thesis(ticker)
    return text
