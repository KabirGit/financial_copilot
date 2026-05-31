# 📊 Financial Research Copilot

An AI-powered research assistant that answers financial analysis questions using real SEC filings (10-K, 10-Q) with grounded retrieval and agentic reasoning.

> "What was Apple's revenue in 2024?" → **$391.0 billion** (cited from 10-K, FY2024)

## Features

- **Grounded answers** — Every response backed by actual SEC filing excerpts, never hallucinated
- **Hybrid retrieval** — Combines vector search (Qdrant) + keyword search (BM25) for precision
- **Intent classification** — Automatically routes queries to specialized tools (compare, risk, metric, thesis, calculate)
- **Table-aware chunking** — Financial tables preserved as structured data, never split mid-row
- **Confidence scoring** — Each answer includes a reliability score (0.0–1.0)
- **Dual interface** — REST API + Streamlit web UI
- **Fast local classification** — Regex-based intent detection skips LLM for common queries

## Architecture

```
User Query → Intent Classification → Retrieval → LLM Synthesis → Response
                    │                      │              │
              (local regex)          (Qdrant + BM25)  (OpenRouter)
```

**Stack:** Python · FastAPI · Streamlit · LangGraph · Qdrant · Sentence-Transformers · OpenRouter

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (for Qdrant vector database)
- [OpenRouter API key](https://openrouter.ai/keys) (free)

### 1. Clone & Install

```bash
git clone https://github.com/your-username/financial-research-copilot.git
cd financial-research-copilot
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OpenRouter API key
```

### 3. Start Qdrant

```bash
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
```

### 4. Ingest SEC Filings

```bash
python -m ingestion.pipeline
```

### 5. Build Search Index

```bash
python -m rag.indexer
```

### 6. Launch

```bash
python launch.py
```

Services will be available at:
- **API:** http://localhost:8080/docs
- **UI:** http://localhost:8501

## Docker Compose (Recommended)

```bash
# Start all services (Qdrant + API + UI)
docker-compose up --build

# Index chunks (first time only, in a separate terminal)
docker-compose exec api python -m ingestion.pipeline
docker-compose exec api python -m rag.indexer
```

## API Usage

```bash
# Health check
curl http://localhost:8080/health

# Research query
curl -X POST http://localhost:8080/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What was Apple revenue in 2024?"}'
```

### Response Format

```json
{
  "answer": "Apple's total revenue for FY2024 was $391.0 billion...",
  "sources": [
    {
      "chunk_id": "AAPL_10K_2024_annual_042",
      "ticker": "AAPL",
      "filing_type": "10-K",
      "fiscal_year": 2024,
      "section": "Financial Data"
    }
  ],
  "confidence": 1.0,
  "tool_calls_made": ["get_financial_metric"],
  "reasoning_trace": "Intent=metric; tools=['get_financial_metric']; ticker=AAPL; confidence=1.0"
}
```

## Supported Query Types

| Intent | Example | Speed |
|--------|---------|-------|
| **metric** | "What was Apple's revenue in 2024?" | ~15s |
| **compare** | "How did margins change Q2 vs Q3?" | ~15s |
| **risk** | "What are Microsoft's risk factors?" | ~15s |
| **thesis** | "Should I invest in Google?" | ~15s |
| **calculate** | "100 * 0.42" | ~2s |
| **general** | "Tell me about Apple's business" | ~15s |

## Supported Tickers

- **AAPL** — Apple Inc.
- **MSFT** — Microsoft Corporation
- **GOOGL** — Alphabet Inc.

(Expandable via `config.py`)

## Project Structure

```
financial_copilot/
├── agent/              # LangGraph agent (graph, tools, prompts, models)
├── api/                # FastAPI REST backend
├── ingestion/          # SEC EDGAR download, parse, chunk pipeline
├── rag/                # Hybrid retrieval (embedder, vector store, BM25)
├── ui/                 # Streamlit web interface
├── deploy/             # Railway & Render deployment configs
├── tests/              # Pytest test suite
├── config.py           # Central configuration
├── launch.py           # Orchestrator (starts API + UI)
├── docker-compose.yml  # Multi-container setup
├── Dockerfile          # Container image
└── requirements.txt    # Python dependencies
```

## Configuration

Key settings in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `OPENROUTER_MODEL` | `google/gemma-4-26b-a4b-it:free` | Primary LLM model |
| `OPENROUTER_FALLBACK_MODEL` | `openai/gpt-oss-20b:free` | Fallback on rate limit |
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Local embedding model (1024-dim) |
| `TOP_K` | 5 | Chunks retrieved per query |
| `HYBRID_ALPHA` | 0.7 | Vector vs BM25 weight (0.7 = 70% vector) |
| `TABLE_BOOST` | 1.3 | Score multiplier for table chunks |

## Running Tests

```bash
pytest tests/ -v
```

## Deployment

### Railway

```bash
railway up
```

### Render

Push to GitHub and connect via Render dashboard. Config in `deploy/render.yaml`.

### Manual Docker

```bash
docker build -t financial-copilot .
docker run -p 8080:8080 --env-file .env financial-copilot
```

## Performance

| Metric | Value |
|--------|-------|
| Cold start (first query) | ~30s |
| Warm queries | ~15s |
| Calculations | ~2s |
| Embedding model | BAAI/bge-large-en-v1.5 (local) |
| LLM | OpenRouter free tier |

## Tech Stack

- **LLM:** OpenRouter (Gemma 4 26B / GPT-OSS-20B free tier)
- **Embeddings:** BAAI/bge-large-en-v1.5 (local, 1024-dim)
- **Vector DB:** Qdrant
- **Keyword Search:** BM25 (rank-bm25)
- **Agent Framework:** LangGraph
- **API:** FastAPI + Uvicorn
- **UI:** Streamlit
- **Data Source:** SEC EDGAR (10-K, 10-Q filings)

## License

MIT
