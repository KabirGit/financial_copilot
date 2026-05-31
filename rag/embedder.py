"""
Local embedding via sentence-transformers (BGE-large).
BGE models use a query prefix for asymmetric retrieval.
Singleton pattern ensures model is loaded only once.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL

# BGE retrieval instruction (documents are embedded without a prefix)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Singleton model instance
_model_instance: SentenceTransformer | None = None


def _get_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    global _model_instance
    if _model_instance is None:
        _model_instance = SentenceTransformer(model_name)
    return _model_instance


class Embedder:
    """Wraps BAAI/bge-large-en-v1.5 for document and query embeddings."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model = _get_model(model_name)

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed passage texts (no query prefix). Returns (n, dim) float32 array."""
        if not texts:
            return np.zeros((0, self._model.get_sentence_embedding_dimension()), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a search query with the BGE instruction prefix."""
        prefixed = BGE_QUERY_PREFIX + query.strip()
        vector = self._model.encode(
            [prefixed],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vector[0], dtype=np.float32)

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())
