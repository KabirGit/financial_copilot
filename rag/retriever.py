"""
Hybrid retriever: dense vectors (Qdrant) + BM25 with metadata filtering.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi
from config import (
    BM25_INDEX_PATH,
    HYBRID_ALPHA,
    TABLE_BOOST,
    TOP_K,
)
from rag.embedder import Embedder
from rag.scoring import min_max_normalize
from rag.filters import (
    RetrievedChunk,
    RetrievalFilters,
    metadata_matches_filters,
)
from rag.qdrant_filters import build_qdrant_filter
from rag.vector_store import QdrantVectorStore

# Re-export for public API
__all__ = ["FinancialRetriever", "RetrievalFilters", "RetrievedChunk"]

# Candidate pool size before fusion (per channel)
CANDIDATE_POOL = 50


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class FinancialRetriever:
    """
    Hybrid search over SEC filing chunks.

    Combines cosine similarity in Qdrant with BM25 sparse scores,
    applies TABLE_BOOST for table chunks, and supports temporal / company filters.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[QdrantVectorStore] = None,
        bm25_index_path: Path = BM25_INDEX_PATH,
    ) -> None:
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or QdrantVectorStore()
        self.bm25_index_path = Path(bm25_index_path)
        self._bm25: Optional[BM25Okapi] = None
        self._corpus_records: list[dict[str, Any]] = []
        self._load_bm25_index()

    def _load_bm25_index(self) -> None:
        path = Path(self.bm25_index_path)
        if not path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {path}. "
                f"Run: python -m rag.indexer  to build it first."
            )
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._corpus_records = data["records"]
        self._bm25 = BM25Okapi(data["tokenized_corpus"])
        logger.info(f"Loaded BM25 index ({len(self._corpus_records)} chunks)")

    def retrieve(
        self,
        query: str,
        filters: Optional[RetrievalFilters] = None,
        top_k: int = TOP_K,
        alpha: float = HYBRID_ALPHA,
    ) -> list[RetrievedChunk]:
        """
        Run hybrid retrieval.

        alpha: weight on vector score (1-alpha on BM25). 0.7 = mostly vector.
        Falls back to BM25-only if Qdrant is unavailable.
        """
        query_vector = None
        vector_scores: dict[str, float] = {}
        hit_payloads: dict[str, dict[str, Any]] = {}

        # Try vector search, gracefully degrade if unavailable
        try:
            if self.vector_store.collection_exists():
                query_vector = self.embedder.embed_query(query)
                qdrant_filter = build_qdrant_filter(filters)
                vector_hits = self.vector_store.search(
                    query_vector,
                    limit=CANDIDATE_POOL,
                    query_filter=qdrant_filter,
                )
                for hit in vector_hits:
                    chunk_id = str(hit.id)
                    vector_scores[chunk_id] = float(hit.score)
                    hit_payloads[chunk_id] = hit.payload or {}
        except Exception as exc:
            logger.warning(f"Qdrant search failed, falling back to BM25-only: {exc}")

        bm25_scores = self._bm25_search(query, filters)
        candidate_ids = set(vector_scores) | set(bm25_scores)
        if not candidate_ids:
            return []

        norm_vector = min_max_normalize({cid: vector_scores.get(cid, 0.0) for cid in candidate_ids})
        norm_bm25 = min_max_normalize({cid: bm25_scores.get(cid, 0.0) for cid in candidate_ids})

        fused: list[tuple[str, float, float, float]] = []
        for chunk_id in candidate_ids:
            v_score = norm_vector[chunk_id]
            b_score = norm_bm25[chunk_id]
            combined = alpha * v_score + (1.0 - alpha) * b_score

            payload = hit_payloads.get(chunk_id)
            if payload is None:
                payload = self._payload_for_chunk_id(chunk_id)
            if payload is None:
                continue

            chunk_type = payload.get("chunk_type", "text")
            if chunk_type == "table":
                combined *= TABLE_BOOST

            fused.append((chunk_id, combined, v_score, b_score))

        fused.sort(key=lambda x: x[1], reverse=True)
        results: list[RetrievedChunk] = []
        for chunk_id, score, v_score, b_score in fused[:top_k]:
            payload = hit_payloads.get(chunk_id) or self._payload_for_chunk_id(chunk_id) or {}
            metadata = {k: v for k, v in payload.items() if k not in ("text", "token_count")}
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=payload.get("text", ""),
                    metadata=metadata,
                    score=score,
                    vector_score=v_score,
                    bm25_score=b_score,
                    token_count=int(payload.get("token_count", 0)),
                )
            )
        return results

    def _bm25_search(self, query: str, filters: Optional[RetrievalFilters]) -> dict[str, float]:
        if self._bm25 is None or not self._corpus_records:
            return {}

        tokens = _tokenize(query)
        if not tokens:
            return {}

        scores = self._bm25.get_scores(tokens)
        out: dict[str, float] = {}
        for idx, record in enumerate(self._corpus_records):
            meta = record.get("metadata", {})
            if not metadata_matches_filters(meta, filters):
                continue
            raw = float(scores[idx])
            if raw > 0:
                out[record["chunk_id"]] = raw

        if not out:
            return {}

        # Keep top candidates by BM25 for fusion
        ranked = sorted(out.items(), key=lambda x: x[1], reverse=True)[:CANDIDATE_POOL]
        n = len(ranked)
        return {cid: float(n - i) for i, (cid, _) in enumerate(ranked)}

    def _payload_for_chunk_id(self, chunk_id: str) -> Optional[dict[str, Any]]:
        for record in self._corpus_records:
            if record.get("chunk_id") == chunk_id:
                return {
                    "text": record.get("text", ""),
                    "token_count": record.get("token_count", 0),
                    **record.get("metadata", {}),
                }
        return None
