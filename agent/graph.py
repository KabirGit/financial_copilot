"""
LangGraph research agent: rewrite → retrieve → synthesize → output.

Performance optimizations:
- Singleton LLM instance (avoids re-creating HTTP clients)
- Local regex-based intent classification (skips LLM rewrite for simple queries)
- Reduced context window sent to synthesis (trim to top chunks)
- Concise synthesis prompt to reduce output tokens
"""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from loguru import logger

from agent.models import ResearchResponse, SourceCitation
from agent.prompts import QUERY_REWRITER_PROMPT, SYNTHESIS_PROMPT, SYSTEM_PROMPT
from rag.filters import RetrievalFilters
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL, OPENROUTER_FALLBACK_MODEL

NO_CONTEXT_MSG = "No relevant filings found."

# Maximum context chars to send to synthesis (prevents huge prompts that slow LLM)
MAX_CONTEXT_CHARS = 6000


class AgentState(TypedDict):
    original_query: str
    rewritten_query: str
    ticker: str | None
    fiscal_year: int | None
    fiscal_quarter: str | None
    intent: str
    expression: str
    retrieved_context: str
    tool_calls_made: list[str]
    raw_chunks: list[dict]
    final_answer: str
    sources: list[dict]
    confidence: float
    reasoning_trace: str


# Singleton LLM instances — avoids re-creating HTTP client on every call
_llm_instance: ChatOpenAI | None = None
_llm_fallback: ChatOpenAI | None = None


def _get_llm(fallback: bool = False) -> ChatOpenAI:
    global _llm_instance, _llm_fallback
    if not fallback:
        if _llm_instance is None:
            _llm_instance = ChatOpenAI(
                model=OPENROUTER_MODEL,
                openai_api_key=OPENROUTER_API_KEY or None,
                openai_api_base=OPENROUTER_BASE_URL,
                temperature=0.2,
                max_tokens=1024,
                default_headers={
                    "HTTP-Referer": "http://localhost:8080",
                    "X-Title": "Financial Research Copilot",
                },
            )
        return _llm_instance
    else:
        if _llm_fallback is None:
            _llm_fallback = ChatOpenAI(
                model=OPENROUTER_FALLBACK_MODEL,
                openai_api_key=OPENROUTER_API_KEY or None,
                openai_api_base=OPENROUTER_BASE_URL,
                temperature=0.2,
                max_tokens=1024,
                default_headers={
                    "HTTP-Referer": "http://localhost:8080",
                    "X-Title": "Financial Research Copilot",
                },
            )
        return _llm_fallback


def _invoke_llm(prompt: str, user_content: str) -> str:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt.format(query=user_content) if "{query}" in prompt else prompt),
    ]
    # Try primary model, fall back on rate limit
    for attempt, use_fallback in enumerate([False, True]):
        try:
            llm = _get_llm(fallback=use_fallback)
            response = llm.invoke(messages)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str and not use_fallback:
                logger.warning(f"Primary model rate-limited, trying fallback...")
                continue
            logger.error(f"LLM call failed: {exc}")
            return ""
    return ""


def _parse_rewrite_json(raw: str, original_query: str) -> dict[str, Any]:
    """Parse rewriter JSON with fallback to general intent."""
    default = {
        "rewritten_query": original_query,
        "ticker": None,
        "fiscal_year": None,
        "fiscal_quarter": None,
        "intent": "general",
        "expression": "",
    }
    if not raw:
        return default

    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning(f"Malformed rewriter JSON: {exc}")
        return default

    intent = str(data.get("intent", "general")).lower()
    if intent not in ("compare", "risk", "metric", "thesis", "calculate", "general"):
        intent = "general"

    expression = data.get("expression") or ""
    if expression is not None:
        expression = str(expression).strip()
    else:
        expression = ""

    ticker = data.get("ticker")
    if ticker:
        ticker = str(ticker).upper()

    fiscal_year = data.get("fiscal_year")
    if fiscal_year is not None:
        try:
            fiscal_year = int(fiscal_year)
        except (TypeError, ValueError):
            fiscal_year = None

    fq = data.get("fiscal_quarter")
    if fq:
        fq = str(fq).upper()

    return {
        "rewritten_query": data.get("rewritten_query") or original_query,
        "ticker": ticker,
        "fiscal_year": fiscal_year,
        "fiscal_quarter": fq,
        "intent": intent,
        "expression": expression,
    }


def _looks_like_arithmetic(query: str) -> bool:
    """Returns True if query is primarily a math expression."""
    stripped = query.strip()
    if re.match(r'^[\d\s\.\+\-\*\/\(\)\%]+$', stripped):
        return True
    if re.search(r'\d+\s*[\+\-\*\/]\s*\d+', stripped):
        return True
    return False


# --- Fast local intent classification (avoids LLM round-trip) ---

_TICKER_RE = re.compile(r'\b(AAPL|MSFT|GOOGL|GOOG|AMZN|TSLA|META|NVDA|APPLE|MICROSOFT|GOOGLE|ALPHABET)\b', re.IGNORECASE)
_YEAR_RE = re.compile(r'\b(20[12]\d)\b')
_QUARTER_RE = re.compile(r'\b(Q[1-4])\b', re.IGNORECASE)
_COMPARE_KEYWORDS = re.compile(
    r'\b(compare|vs\.?|versus|change|differ|between.*and|Q\d\s*(vs|and|to)\s*Q\d)\b', re.IGNORECASE
)
_RISK_KEYWORDS = re.compile(r'\b(risk\s*factor|risks?|threat|headwind)\b', re.IGNORECASE)
_THESIS_KEYWORDS = re.compile(r'\b(invest|thesis|bull|bear|buy|sell|hold|outlook)\b', re.IGNORECASE)
_METRIC_KEYWORDS = re.compile(
    r'\b(revenue|income|margin|earnings|eps|ebitda|cash\s*flow|sales|profit|debt|assets|liabilities|'
    r'stock|share\s*price|perform|return|growth|dividend|valuation|market\s*cap)\b',
    re.IGNORECASE,
)


def _local_classify(query: str) -> dict[str, Any] | None:
    """
    Fast regex-based intent classification. Returns parsed dict if confident,
    None if the query is ambiguous and needs LLM rewrite.
    """
    if _looks_like_arithmetic(query):
        return {
            "rewritten_query": query,
            "ticker": None,
            "fiscal_year": None,
            "fiscal_quarter": None,
            "intent": "calculate",
            "expression": query.strip(),
        }

    ticker_match = _TICKER_RE.search(query)
    ticker = None
    if ticker_match:
        raw_ticker = ticker_match.group(0).upper()
        # Map company names to tickers
        name_to_ticker = {
            "APPLE": "AAPL", "MICROSOFT": "MSFT",
            "GOOGLE": "GOOGL", "ALPHABET": "GOOGL", "GOOG": "GOOGL",
        }
        ticker = name_to_ticker.get(raw_ticker, raw_ticker)

    year_match = _YEAR_RE.search(query)
    fiscal_year = int(year_match.group(0)) if year_match else None

    quarter_match = _QUARTER_RE.search(query)
    fiscal_quarter = quarter_match.group(0).upper() if quarter_match else None

    # Determine intent
    if _COMPARE_KEYWORDS.search(query):
        intent = "compare"
    elif _RISK_KEYWORDS.search(query):
        intent = "risk"
    elif _THESIS_KEYWORDS.search(query):
        intent = "thesis"
    elif _METRIC_KEYWORDS.search(query):
        intent = "metric"
    else:
        # Ambiguous — fall back to LLM only if we can't determine intent
        return None

    return {
        "rewritten_query": query,
        "ticker": ticker,
        "fiscal_year": fiscal_year,
        "fiscal_quarter": fiscal_quarter,
        "intent": intent,
        "expression": "",
    }


def rewrite_node(state: AgentState) -> dict[str, Any]:
    query = state["original_query"]

    # Try fast local classification first (no LLM call needed)
    local_result = _local_classify(query)
    if local_result is not None:
        logger.info(f"Fast classify: intent={local_result['intent']}, ticker={local_result['ticker']}")
        return local_result

    # Fall back to LLM for ambiguous queries
    raw = _invoke_llm(QUERY_REWRITER_PROMPT, query)
    parsed = _parse_rewrite_json(raw, query)
    result = {
        "rewritten_query": parsed["rewritten_query"],
        "ticker": parsed["ticker"],
        "fiscal_year": parsed["fiscal_year"],
        "fiscal_quarter": parsed["fiscal_quarter"],
        "intent": parsed["intent"],
        "expression": parsed.get("expression", ""),
    }
    if _looks_like_arithmetic(query):
        result["intent"] = "calculate"
        result["expression"] = query.strip()
    return result


def _extract_compare_periods(query: str, fiscal_year: int | None) -> tuple[str, str, str]:
    """Infer period1, period2, and metric hint from the user query."""
    year = fiscal_year or 2024
    quarters = re.findall(r"\bQ([1-4])\b", query, re.IGNORECASE)
    if len(quarters) >= 2:
        p1 = f"{year}_Q{quarters[0]}"
        p2 = f"{year}_Q{quarters[1]}"
    else:
        p1, p2 = f"{year}_Q2", f"{year}_Q3"

    metric = "gross margin"
    for term in ("gross margin", "revenue", "earnings", "operating income", "net income"):
        if term in query.lower():
            metric = term
            break
    return p1, p2, metric


def retrieval_node(state: AgentState) -> dict[str, Any]:
    from agent.tools import (
        CHUNK_SEPARATOR,
        HybridRetriever,
        _format_chunk,
        calculate_metric,
        chunks_to_raw_dicts,
        run_compare_quarters,
        run_generate_investment_thesis,
        run_get_financial_metric,
        run_get_risk_factors,
    )

    intent = state.get("intent", "general")
    ticker = state.get("ticker") or "AAPL"
    fiscal_year = state.get("fiscal_year") or 2024
    fiscal_quarter = state.get("fiscal_quarter")
    rewritten = state.get("rewritten_query", state["original_query"])
    tool_calls: list[str] = []
    context = ""
    raw_chunks: list[dict] = []

    try:
        if intent == "calculate":
            tool_calls.append("calculate_metric")
            expression = state.get("expression", "") or rewritten
            context = calculate_metric.invoke({"expression": expression})
            raw_chunks = []
        elif intent == "compare":
            tool_calls.append("compare_quarters")
            p1, p2, metric = _extract_compare_periods(
                state["original_query"], state.get("fiscal_year")
            )
            context, raw_chunks = run_compare_quarters(ticker, metric, p1, p2)
        elif intent == "risk":
            tool_calls.append("get_risk_factors")
            context, raw_chunks = run_get_risk_factors(ticker, fiscal_year)
        elif intent == "metric":
            tool_calls.append("get_financial_metric")
            context, raw_chunks = run_get_financial_metric(
                ticker, rewritten, fiscal_year, fiscal_quarter
            )
        elif intent == "thesis":
            tool_calls.append("generate_investment_thesis")
            context, raw_chunks = run_generate_investment_thesis(ticker)
        else:
            tool_calls.append("retrieve")
            try:
                retriever = HybridRetriever()
                filters = None
                if ticker:
                    filters = RetrievalFilters(
                        ticker=ticker,
                        fiscal_year=fiscal_year,
                        fiscal_quarter=fiscal_quarter,
                    )
                chunks = retriever.retrieve(rewritten, filters=filters)
                if chunks:
                    context = CHUNK_SEPARATOR.join(_format_chunk(c) for c in chunks)
                    raw_chunks = chunks_to_raw_dicts(chunks)
                else:
                    context = NO_CONTEXT_MSG
            except Exception as exc:
                logger.error(f"General retrieval failed: {exc}")
                context = NO_CONTEXT_MSG
    except Exception as exc:
        logger.error(f"Retrieval node failed: {exc}")
        context = NO_CONTEXT_MSG

    if intent == "calculate" and context and "error" not in context.lower():
        confidence = 1.0
    elif not context or context.strip() == "" or "No relevant" in context and len(context) < 120:
        confidence = 0.4
        if not context:
            context = NO_CONTEXT_MSG
    elif len(context) > 800:
        confidence = 1.0
    else:
        confidence = 0.7

    return {
        "retrieved_context": context,
        "tool_calls_made": tool_calls,
        "raw_chunks": raw_chunks,
        "confidence": confidence,
    }


def synthesis_node(state: AgentState) -> dict[str, Any]:
    context = state.get("retrieved_context", NO_CONTEXT_MSG)
    intent = state.get("intent", "general")

    # Skip LLM for calculations — the answer is already computed
    if intent == "calculate":
        confidence = state.get("confidence", 1.0)
        trace = f"Intent=calculate; tools={state.get('tool_calls_made')}; confidence={confidence:.1f}"
        return {
            "final_answer": f"**Result:** {context}",
            "confidence": confidence,
            "reasoning_trace": trace,
        }

    # Trim context to avoid sending huge prompts (major speed improvement)
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n\n[... additional context truncated for speed ...]"

    prompt = SYNTHESIS_PROMPT.format(
        query=state["original_query"],
        context=context,
    )

    try:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        # Try primary, fallback on rate limit
        final = None
        for use_fallback in [False, True]:
            try:
                llm = _get_llm(fallback=use_fallback)
                answer = llm.invoke(messages)
                final = answer.content if hasattr(answer, "content") else str(answer)
                break
            except Exception as exc:
                if "429" in str(exc) and not use_fallback:
                    logger.warning("Synthesis: primary rate-limited, trying fallback...")
                    continue
                raise
    except Exception as exc:
        logger.error(f"Synthesis failed: {exc}")
        final = (
            "Unable to generate a synthesis due to an API error. "
            f"Retrieved context preview: {context[:500]}..."
        )

    if not final:
        final = "No answer could be generated from the available context."

    confidence = state.get("confidence", 0.7)
    if context == NO_CONTEXT_MSG or not context.strip():
        confidence = min(confidence, 0.4)
    elif len(context) > 800:
        confidence = max(confidence, 1.0)

    trace = (
        f"Intent={state.get('intent')}; tools={state.get('tool_calls_made')}; "
        f"ticker={state.get('ticker')}; confidence={confidence:.1f}"
    )

    return {
        "final_answer": final,
        "confidence": confidence,
        "reasoning_trace": trace,
    }


def output_node(state: AgentState) -> dict[str, Any]:
    """Pass-through; citations are built in state_to_response from raw_chunks."""
    return {}


def build_graph():
    """Compile the linear research graph."""
    workflow = StateGraph(AgentState)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("output", output_node)

    workflow.set_entry_point("rewrite")
    workflow.add_edge("rewrite", "retrieval")
    workflow.add_edge("retrieval", "synthesis")
    workflow.add_edge("synthesis", "output")
    workflow.add_edge("output", END)

    return workflow.compile()


def state_to_response(state: AgentState) -> ResearchResponse:
    """Build ResearchResponse from terminal graph state."""
    raw_chunks = state.get("raw_chunks", [])
    if raw_chunks:
        citations = [
            SourceCitation(
                chunk_id=c.get("chunk_id", ""),
                ticker=c.get("ticker", ""),
                section=c.get("section", ""),
                date_filed=c.get("date_filed", ""),
                filing_type=c.get("filing_type", ""),
                fiscal_year=c.get("fiscal_year"),
                fiscal_quarter=c.get("fiscal_quarter"),
            )
            for c in raw_chunks
        ]
    else:
        from agent.tools import parse_sources_from_context

        parsed = parse_sources_from_context(state.get("retrieved_context", ""))
        citations = [
            SourceCitation(
                chunk_id=s["chunk_id"],
                ticker=s["ticker"],
                section=s["section"],
                date_filed=s["date_filed"],
                filing_type=s["filing_type"],
                fiscal_year=s.get("fiscal_year"),
                fiscal_quarter=s.get("fiscal_quarter"),
            )
            for s in parsed
        ]
    return ResearchResponse(
        answer=state.get("final_answer", ""),
        sources=citations,
        confidence=float(state.get("confidence", 0.4)),
        tool_calls_made=list(state.get("tool_calls_made", [])),
        reasoning_trace=state.get("reasoning_trace", ""),
    )
