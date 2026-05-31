"""
FastAPI endpoint tests — mock run_research; no live Gemini/Qdrant.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from agent.models import ResearchResponse, SourceCitation

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    from api.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def test_health_endpoint(client):
    with patch("api.main._check_qdrant", return_value=True):
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


async def test_tickers_endpoint(client):
    response = await client.get("/tickers")
    assert response.status_code == 200
    data = response.json()
    assert "AAPL" in data["tickers"]


async def test_research_endpoint_success(client):
    mock_response = ResearchResponse(
        answer="Apple gross margins improved in Q3 2024.",
        sources=[
            SourceCitation(
                chunk_id="AAPL_10Q_2024_Q3_001",
                ticker="AAPL",
                section="MD&A",
                date_filed="2024-08-01",
                filing_type="10-Q",
                fiscal_year=2024,
                fiscal_quarter="Q3",
            )
        ],
        confidence=0.9,
        tool_calls_made=["compare_quarters"],
        reasoning_trace="Intent=compare",
    )

    with patch("api.main._run_research_with_timeout", return_value=mock_response):
        response = await client.post(
            "/research",
            json={"query": "Apple gross margin Q3 2024"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["answer"] == mock_response.answer


async def test_research_endpoint_empty_query(client):
    response = await client.post("/research", json={"query": ""})
    assert response.status_code == 422


async def test_root_endpoint(client):
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
