"""
Index all processed chunks: embed → Qdrant, build BM25 sparse index.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any, Iterator

from loguru import logger
from tqdm import tqdm

from config import BM25_INDEX_PATH, DATA_PROCESSED_DIR, TARGET_TICKERS
from rag.embedder import Embedder
from rag.vector_store import QdrantVectorStore

BATCH_SIZE = 32


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def iter_chunk_files(processed_dir: Path | None = None) -> list[Path]:
    root = processed_dir or DATA_PROCESSED_DIR
    paths = sorted(root.glob("*_chunks.jsonl"))
    if not paths:
        logger.warning(f"No chunk files in {root}")
    return paths


def load_chunks_from_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _chunk_to_payload(chunk: dict[str, Any]) -> dict[str, Any]:
    meta = chunk["metadata"]
    return {
        "text": chunk["text"],
        "token_count": chunk["token_count"],
        **meta,
    }


class ChunkIndexer:
    """Embeds JSONL chunks and upserts them into Qdrant + BM25 index."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        vector_store: QdrantVectorStore | None = None,
    ) -> None:
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or QdrantVectorStore()

    def index_all(
        self,
        recreate: bool = False,
        tickers: list[str] | None = None,
        processed_dir: Path | None = None,
    ) -> int:
        """
        Index chunks for all tickers (or a subset). Returns total points indexed.
        """
        self.vector_store.ensure_collection(recreate=recreate)

        files = iter_chunk_files(processed_dir)
        if tickers:
            ticker_set = {t.upper() for t in tickers}
            files = [p for p in files if p.stem.replace("_chunks", "").upper() in ticker_set]

        all_records: list[dict[str, Any]] = []
        total = 0

        for jsonl_path in files:
            ticker = jsonl_path.stem.replace("_chunks", "")
            logger.info(f"Indexing {ticker} from {jsonl_path.name}")
            batch_ids: list[str] = []
            batch_texts: list[str] = []
            batch_payloads: list[dict[str, Any]] = []

            chunks = list(load_chunks_from_jsonl(jsonl_path))
            for chunk in tqdm(chunks, desc=ticker):
                meta = chunk["metadata"]
                chunk_id = meta["chunk_id"]
                text = chunk["text"]

                batch_ids.append(chunk_id)
                batch_texts.append(text)
                batch_payloads.append(_chunk_to_payload(chunk))

                all_records.append(
                    {
                        "chunk_id": chunk_id,
                        "text": text,
                        "token_count": chunk["token_count"],
                        "metadata": meta,
                    }
                )

                if len(batch_ids) >= BATCH_SIZE:
                    self._upsert_batch(batch_ids, batch_texts, batch_payloads)
                    total += len(batch_ids)
                    batch_ids, batch_texts, batch_payloads = [], [], []

            if batch_ids:
                self._upsert_batch(batch_ids, batch_texts, batch_payloads)
                total += len(batch_ids)

        self._save_bm25_index(all_records)
        logger.info(f"Indexed {total} chunks into {self.vector_store.collection}")
        return total

    def _upsert_batch(
        self,
        chunk_ids: list[str],
        texts: list[str],
        payloads: list[dict[str, Any]],
    ) -> None:
        vectors = self.embedder.embed_documents(texts, batch_size=BATCH_SIZE)
        self.vector_store.upsert_batch(chunk_ids, vectors.tolist(), payloads)

    def _save_bm25_index(self, records: list[dict[str, Any]]) -> None:
        tokenized = [_tokenize(r["text"]) for r in records]
        # BM25Okapi requires non-empty token lists; use a placeholder token for empty docs
        tokenized = [t if t else ["_empty_"] for t in tokenized]

        BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump({"records": records, "tokenized_corpus": tokenized}, f)
        logger.info(f"Saved BM25 index ({len(records)} docs) to {BM25_INDEX_PATH}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Index financial chunks into Qdrant")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the Qdrant collection",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help=f"Subset of tickers (default: {TARGET_TICKERS})",
    )
    args = parser.parse_args()

    indexer = ChunkIndexer()
    tickers = args.tickers or TARGET_TICKERS
    indexer.index_all(recreate=args.recreate, tickers=tickers)


if __name__ == "__main__":
    main()
