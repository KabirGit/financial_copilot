"""Qdrant filter construction (no numpy dependency)."""

from __future__ import annotations

from typing import Optional

from qdrant_client.http import models as qm

from rag.filters import RetrievalFilters


def build_qdrant_filter(filters: Optional[RetrievalFilters]) -> Optional[qm.Filter]:
    """Translate RetrievalFilters into a Qdrant Filter (None = no filter)."""
    if filters is None:
        return None

    must: list[qm.FieldCondition] = []

    if filters.ticker:
        must.append(qm.FieldCondition(key="ticker", match=qm.MatchValue(value=filters.ticker)))
    if filters.company:
        must.append(qm.FieldCondition(key="company", match=qm.MatchValue(value=filters.company)))
    if filters.filing_type:
        must.append(
            qm.FieldCondition(key="filing_type", match=qm.MatchValue(value=filters.filing_type))
        )
    if filters.fiscal_quarter:
        must.append(
            qm.FieldCondition(
                key="fiscal_quarter", match=qm.MatchValue(value=filters.fiscal_quarter)
            )
        )
    if filters.section:
        must.append(qm.FieldCondition(key="section", match=qm.MatchValue(value=filters.section)))
    if filters.chunk_type:
        must.append(
            qm.FieldCondition(key="chunk_type", match=qm.MatchValue(value=filters.chunk_type))
        )
    if filters.fiscal_year is not None:
        must.append(
            qm.FieldCondition(key="fiscal_year", match=qm.MatchValue(value=filters.fiscal_year))
        )
    elif filters.fiscal_year_min is not None or filters.fiscal_year_max is not None:
        must.append(
            qm.FieldCondition(
                key="fiscal_year",
                range=qm.Range(
                    gte=filters.fiscal_year_min,
                    lte=filters.fiscal_year_max,
                ),
            )
        )
    if filters.date_filed_after or filters.date_filed_before:
        must.append(
            qm.FieldCondition(
                key="date_filed",
                range=qm.Range(
                    gte=filters.date_filed_after,
                    lte=filters.date_filed_before,
                ),
            )
        )

    if not must:
        return None
    return qm.Filter(must=must)
