"""Retrieval filter and result types shared across vector store and retriever."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class RetrievalFilters:
    """Company- and time-aware filters applied to vector and BM25 search."""

    ticker: Optional[str] = None
    company: Optional[str] = None
    filing_type: Optional[str] = None
    fiscal_year: Optional[int] = None
    fiscal_year_min: Optional[int] = None
    fiscal_year_max: Optional[int] = None
    fiscal_quarter: Optional[str] = None
    section: Optional[str] = None
    chunk_type: Optional[str] = None
    date_filed_after: Optional[str] = None  # ISO date, inclusive
    date_filed_before: Optional[str] = None  # ISO date, inclusive


@dataclass
class RetrievedChunk:
    """A ranked chunk returned by hybrid retrieval."""

    text: str
    metadata: dict[str, Any]
    score: float
    vector_score: float = 0.0
    bm25_score: float = 0.0
    token_count: int = 0
    chunk_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "metadata": self.metadata,
            "score": self.score,
            "vector_score": self.vector_score,
            "bm25_score": self.bm25_score,
            "token_count": self.token_count,
            "chunk_id": self.chunk_id,
        }


def metadata_matches_filters(meta: dict[str, Any], filters: Optional[RetrievalFilters]) -> bool:
    """Apply the same constraints as Qdrant filters to in-memory BM25 candidates."""
    if filters is None:
        return True

    if filters.ticker and meta.get("ticker") != filters.ticker:
        return False
    if filters.company and meta.get("company") != filters.company:
        return False
    if filters.filing_type and meta.get("filing_type") != filters.filing_type:
        return False
    if filters.fiscal_quarter and meta.get("fiscal_quarter") != filters.fiscal_quarter:
        return False
    if filters.section and meta.get("section") != filters.section:
        return False
    if filters.chunk_type and meta.get("chunk_type") != filters.chunk_type:
        return False

    year = meta.get("fiscal_year")
    if filters.fiscal_year is not None and year != filters.fiscal_year:
        return False
    if filters.fiscal_year_min is not None and (year is None or year < filters.fiscal_year_min):
        return False
    if filters.fiscal_year_max is not None and (year is None or year > filters.fiscal_year_max):
        return False

    date_filed = meta.get("date_filed") or ""
    if filters.date_filed_after and date_filed < filters.date_filed_after:
        return False
    if filters.date_filed_before and date_filed > filters.date_filed_before:
        return False

    return True
