"""Pydantic response models for the research agent."""

from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    chunk_id: str
    ticker: str
    section: str
    date_filed: str
    filing_type: str
    fiscal_year: int | None
    fiscal_quarter: str | None


class ResearchResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    confidence: float = Field(ge=0.0, le=1.0)
    tool_calls_made: list[str]
    reasoning_trace: str
