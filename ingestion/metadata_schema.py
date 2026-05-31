"""
Defines the metadata schema for every chunk produced by this pipeline.
Every downstream system (vector store, retriever, agent) depends on this schema.
Do not change field names without updating all consumers.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Literal
import json


@dataclass
class ChunkMetadata:
    # Identity
    chunk_id: str                          # unique: "{ticker}_{filing_type}_{year}_{quarter}_{idx}"
    source_file: str                       # original filename

    # Company
    company: str                           # e.g. "Apple Inc."
    ticker: str                            # e.g. "AAPL"

    # Filing info
    filing_type: Literal["10-K", "10-Q", "8-K", "transcript", "news"]
    fiscal_year: int                       # e.g. 2024
    fiscal_quarter: Optional[str]          # e.g. "Q1", "Q2", None for annual

    # Document position
    section: str                           # e.g. "Risk Factors", "MD&A", "Financial Statements"
    page_number: Optional[int]
    chunk_index: int                       # position within this document

    # Content type — critical for table-aware RAG
    chunk_type: Literal["text", "table", "header"]

    # Date filed (ISO format string)
    date_filed: str                        # e.g. "2024-02-01"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class Chunk:
    """A text chunk with its metadata envelope."""
    text: str
    metadata: ChunkMetadata
    token_count: int

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "token_count": self.token_count,
            "metadata": self.metadata.to_dict()
        }