"""
Unit tests for hybrid retrieval helpers (no Qdrant / embedding model required).
"""

import pytest

from rag.filters import RetrievalFilters, metadata_matches_filters
from rag.scoring import min_max_normalize
from rag.qdrant_filters import build_qdrant_filter


def _sample_meta(**overrides):
    base = {
        "chunk_id": "AAPL_10K_2024_annual_0001",
        "ticker": "AAPL",
        "company": "Apple Inc.",
        "filing_type": "10-K",
        "fiscal_year": 2024,
        "fiscal_quarter": None,
        "section": "Risk Factors",
        "chunk_type": "text",
        "date_filed": "2024-09-30",
    }
    base.update(overrides)
    return base


class TestMetadataFilters:
    def test_ticker_filter(self):
        f = RetrievalFilters(ticker="AAPL")
        assert metadata_matches_filters(_sample_meta(), f)
        assert not metadata_matches_filters(_sample_meta(ticker="MSFT"), f)

    def test_fiscal_year_range(self):
        f = RetrievalFilters(fiscal_year_min=2023, fiscal_year_max=2024)
        assert metadata_matches_filters(_sample_meta(fiscal_year=2024), f)
        assert not metadata_matches_filters(_sample_meta(fiscal_year=2022), f)

    def test_date_range(self):
        f = RetrievalFilters(date_filed_after="2024-01-01", date_filed_before="2024-12-31")
        assert metadata_matches_filters(_sample_meta(date_filed="2024-06-15"), f)
        assert not metadata_matches_filters(_sample_meta(date_filed="2023-12-31"), f)

    def test_no_filter_passes_all(self):
        assert metadata_matches_filters(_sample_meta(), None)


class TestQdrantFilterBuilder:
    def test_empty_filters_returns_none(self):
        assert build_qdrant_filter(None) is None
        assert build_qdrant_filter(RetrievalFilters()) is None

    def test_ticker_condition(self):
        qfilter = build_qdrant_filter(RetrievalFilters(ticker="MSFT"))
        assert qfilter is not None
        assert len(qfilter.must) == 1
        assert qfilter.must[0].key == "ticker"

    def test_fiscal_year_range(self):
        qfilter = build_qdrant_filter(
            RetrievalFilters(fiscal_year_min=2022, fiscal_year_max=2024)
        )
        assert qfilter is not None
        year_cond = next(c for c in qfilter.must if c.key == "fiscal_year")
        assert year_cond.range.gte == 2022
        assert year_cond.range.lte == 2024


class TestScoreNormalization:
    def test_min_max_normalize(self):
        scores = {"a": 1.0, "b": 3.0, "c": 2.0}
        norm = min_max_normalize(scores)
        assert norm["a"] == pytest.approx(0.0)
        assert norm["b"] == pytest.approx(1.0)
        assert norm["c"] == pytest.approx(0.5)

    def test_flat_scores_all_ones(self):
        norm = min_max_normalize({"x": 2.0, "y": 2.0})
        assert norm["x"] == 1.0
        assert norm["y"] == 1.0


class TestTableBoostLogic:
    def test_table_chunk_type_in_metadata(self):
        meta = _sample_meta(chunk_type="table")
        assert meta["chunk_type"] == "table"
