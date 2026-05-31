"""
FastAPI backend for the Financial Research Copilot.
"""

from __future__ import annotations

import concurrent.futures
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

load_dotenv()

from agent.runner import run_research
from agent.models import ResearchResponse
from config import BM25_INDEX_PATH, OPENROUTER_MODEL, QDRANT_HOST, QDRANT_PORT, TARGET_TICKERS

RESEARCH_TIMEOUT_SECONDS = 90


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    if not Path(BM25_INDEX_PATH).exists():
        logging.warning(
            "WARNING: BM25 index missing at %s. Run: python -m rag.indexer before serving requests.",
            BM25_INDEX_PATH,
        )
    yield
    # Shutdown (nothing to clean up)


app = FastAPI(
    title="Financial Research Copilot API",
    version="1.0.0",
    docs_url="/docs",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    query: str = Field(min_length=1)
    ticker: str | None = None
    fiscal_year: int | None = None


def _check_qdrant() -> bool:
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        client.get_collections()
        return True
    except Exception as exc:
        logger.warning(f"Qdrant health check failed: {exc}")
        return False


def _run_research_with_timeout(query: str, timeout: int = RESEARCH_TIMEOUT_SECONDS) -> ResearchResponse:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_research, query)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            raise TimeoutError(f"Research timed out after {timeout}s") from exc


@app.get("/")
async def root() -> dict:
    return {
        "message": "Financial Research Copilot API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "qdrant": _check_qdrant(),
        "model": OPENROUTER_MODEL,
    }


@app.get("/tickers")
async def tickers() -> dict:
    return {"tickers": list(TARGET_TICKERS)}


@app.post("/research")
async def research(request: ResearchRequest) -> ResearchResponse:
    # ticker / fiscal_year reserved for future override support
    try:
        return _run_research_with_timeout(request.query)
    except TimeoutError:
        return JSONResponse(
            status_code=504,
            content={"error": "Research timed out", "query": request.query},
        )
    except Exception as exc:
        logger.error(f"Research failed: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "query": request.query},
        )
