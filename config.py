from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Tickers to ingest — start small, expandable later
TARGET_TICKERS = ["AAPL", "MSFT", "GOOGL"]

# Filing types to download
FILING_TYPES = ["10-K", "10-Q"]

# How many of each filing type per ticker
FILINGS_PER_TYPE = 4

# Chunking config
CHUNK_SIZE = 512        # tokens
CHUNK_OVERLAP = 64      # tokens

# Minimum chunk length to keep (filter noise)
MIN_CHUNK_TOKENS = 40

# Qdrant config (configurable via env for Docker)
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = "financial_copilot"

# Embedding model (free, runs locally via sentence-transformers)
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = 1024  # BGE-large output dimension

# Retrieval config
TOP_K = 5                    # chunks to retrieve per query (reduced for speed)
HYBRID_ALPHA = 0.7           # 0=pure BM25, 1=pure vector, 0.7=mostly vector
TABLE_BOOST = 1.3            # score multiplier for table chunks (they contain numbers)

# Gemini config (legacy, kept for reference)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"  # free tier model

# OpenRouter config (primary LLM provider)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "google/gemma-4-26b-a4b-it:free"
# Fallback model if primary is rate-limited
OPENROUTER_FALLBACK_MODEL = "openai/gpt-oss-20b:free"

# BM25 sparse index (built during indexing)
BM25_INDEX_PATH = DATA_PROCESSED_DIR / "bm25_index.pkl"