"""
Qdrant vector store with payload indexes for financial metadata filtering.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional, Sequence

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from config import (
    EMBEDDING_DIM,
    QDRANT_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
)
from rag.qdrant_filters import build_qdrant_filter

# Re-export for convenience
__all__ = ["QdrantVectorStore", "build_qdrant_filter"]

PAYLOAD_INDEX_FIELDS = {
    "ticker": qm.PayloadSchemaType.KEYWORD,
    "company": qm.PayloadSchemaType.KEYWORD,
    "filing_type": qm.PayloadSchemaType.KEYWORD,
    "fiscal_year": qm.PayloadSchemaType.INTEGER,
    "fiscal_quarter": qm.PayloadSchemaType.KEYWORD,
    "section": qm.PayloadSchemaType.KEYWORD,
    "chunk_type": qm.PayloadSchemaType.KEYWORD,
    "date_filed": qm.PayloadSchemaType.KEYWORD,
    "chunk_id": qm.PayloadSchemaType.KEYWORD,
}


class QdrantVectorStore:
    """Manages the financial_copilot Qdrant collection."""

    def __init__(
        self,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
        collection: str = QDRANT_COLLECTION,
    ) -> None:
        self.collection = collection
        self.client = QdrantClient(host=host, port=port)

    def collection_exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def ensure_collection(self, recreate: bool = False) -> None:
        if recreate and self.collection_exists():
            logger.warning(f"Recreating collection {self.collection}")
            self.client.delete_collection(self.collection)

        if not self.collection_exists():
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=qm.Distance.COSINE,
                ),
            )
            for field, schema in PAYLOAD_INDEX_FIELDS.items():
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=schema,
                )
            logger.info(f"Created collection {self.collection}")

    def upsert_batch(
        self,
        chunk_ids: list[str],
        vectors: Sequence[Sequence[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        _NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # UUID namespace
        points = [
            qm.PointStruct(
                id=str(uuid.uuid5(_NS, chunk_ids[i])),  # stable UUID from string ID
                vector=list(vectors[i]),
                payload=payloads[i],
            )
            for i in range(len(chunk_ids))
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(
        self,
        query_vector: Sequence[float],
        limit: int,
        query_filter: Optional[qm.Filter] = None,
    ) -> list[qm.ScoredPoint]:
        return self.client.search(
            collection_name=self.collection,
            query_vector=list(query_vector),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

    def count_points(self) -> int:
        info = self.client.get_collection(self.collection)
        return int(info.points_count or 0)
