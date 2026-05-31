"""
Splits parsed sections into token-sized chunks with full metadata.
Uses tiktoken for accurate token counting (same tokenizer as GPT/Gemini).

Key design: table chunks are NEVER split. A table that fits within 2x 
CHUNK_SIZE is kept whole. Only text sections are split.
"""

import tiktoken
import uuid
from pathlib import Path
from loguru import logger
from ingestion.metadata_schema import Chunk, ChunkMetadata
from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_TOKENS
import re


class Chunker:
    def __init__(self):
        # cl100k_base is the tokenizer for GPT-4, text-embedding-3-small, 
        # and compatible with Gemini token estimates
        self.enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.enc.encode(text))

    def chunk_document(
        self,
        sections: list,
        ticker: str,
        filing_type: str,
        fiscal_year: int,
        fiscal_quarter: str | None,
        date_filed: str,
        company: str,
        source_file: str
    ) -> list:
        """
        Takes parsed sections from FilingParser and produces Chunk objects.
        Tables are kept whole. Text is split by token count with overlap.
        """
        chunks = []
        chunk_index = 0

        for section in sections:
            content = section.get("content", "").strip()
            section_name = section.get("section_name", "Unknown")
            content_type = section.get("content_type", "text")
            page_number = section.get("page_number")

            if not content or self.count_tokens(content) < MIN_CHUNK_TOKENS:
                continue

            if content_type == "table":
                # Tables: keep whole (up to 2x chunk size), never split
                token_count = self.count_tokens(content)
                if token_count > CHUNK_SIZE * 2:
                    # Table is too large — split at row boundaries
                    sub_chunks = self._split_table(content)
                else:
                    sub_chunks = [content]

                for sub in sub_chunks:
                    tc = self.count_tokens(sub)
                    if tc < MIN_CHUNK_TOKENS:
                        continue
                    meta = ChunkMetadata(
                        chunk_id=self._make_id(ticker, filing_type, fiscal_year, fiscal_quarter, chunk_index),
                        source_file=source_file,
                        company=company,
                        ticker=ticker,
                        filing_type=filing_type,
                        fiscal_year=fiscal_year,
                        fiscal_quarter=fiscal_quarter,
                        section=section_name,
                        page_number=page_number,
                        chunk_index=chunk_index,
                        chunk_type="table",
                        date_filed=date_filed,
                    )
                    chunks.append(Chunk(text=sub, metadata=meta, token_count=tc))
                    chunk_index += 1

            else:
                # Text: split into overlapping windows
                text_chunks = self._split_text(content)
                for text_chunk in text_chunks:
                    tc = self.count_tokens(text_chunk)
                    if tc < MIN_CHUNK_TOKENS:
                        continue
                    meta = ChunkMetadata(
                        chunk_id=self._make_id(ticker, filing_type, fiscal_year, fiscal_quarter, chunk_index),
                        source_file=source_file,
                        company=company,
                        ticker=ticker,
                        filing_type=filing_type,
                        fiscal_year=fiscal_year,
                        fiscal_quarter=fiscal_quarter,
                        section=section_name,
                        page_number=page_number,
                        chunk_index=chunk_index,
                        chunk_type="text",
                        date_filed=date_filed,
                    )
                    chunks.append(Chunk(text=text_chunk, metadata=meta, token_count=tc))
                    chunk_index += 1

        return chunks

    def _split_text(self, text: str) -> list:
        """
        Split text into token-bounded chunks with overlap.
        Tries to split at sentence boundaries first.
        """
        tokens = self.enc.encode(text)
        if len(tokens) <= CHUNK_SIZE:
            return [text]

        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + CHUNK_SIZE, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.enc.decode(chunk_tokens)
            chunks.append(chunk_text)
            if end == len(tokens):
                break
            start = end - CHUNK_OVERLAP

        return chunks

    def _split_table(self, table_md: str) -> list:
        """
        Split an oversized markdown table into sub-tables.
        Preserves header row in each sub-table.
        """
        lines = table_md.split("\n")
        if len(lines) < 3:
            return [table_md]

        header = lines[0]
        separator = lines[1]
        data_rows = lines[2:]

        sub_tables = []
        # ~20 rows per sub-table
        chunk_size = 20
        for i in range(0, len(data_rows), chunk_size):
            batch = data_rows[i:i + chunk_size]
            sub_table = "\n".join([header, separator] + batch)
            sub_tables.append(sub_table)

        return sub_tables

    def _make_id(self, ticker, filing_type, year, quarter, idx) -> str:
        q = quarter or "annual"
        filing_clean = filing_type.replace("-", "")
        return f"{ticker}_{filing_clean}_{year}_{q}_{idx:04d}"