"""
Orchestrates the full ingestion pipeline:
  1. Download filings from SEC EDGAR
  2. Parse each filing document  
  3. Chunk into metadata-tagged pieces
  4. Save to data/processed/ as .jsonl files

Run this file directly: python -m ingestion.pipeline
One .jsonl file is produced per ticker. Each line is one Chunk as JSON.
"""

import json
import re
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from ingestion.sec_downloader import SECDownloader
from ingestion.pdf_parser import FilingParser
from ingestion.chunker import Chunker
from config import DATA_PROCESSED_DIR, TARGET_TICKERS


# Map of ticker to company full name
TICKER_TO_COMPANY = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "TSLA": "Tesla Inc.",
    "META": "Meta Platforms Inc.",
    "NVDA": "NVIDIA Corporation",
}


def estimate_fiscal_info(filing_dir_name: str, filing_type: str) -> tuple:
    """
    Extract year, quarter, and date from the filing directory name.
    SEC EDGAR folder names are accession numbers like: 0000320193-22-000108
    The second segment is the year (2 digits), we need to convert to 4 digits.
    Returns: (fiscal_year, fiscal_quarter, date_filed)
    """
    # Accession number format: CIK-YY-NNNNNN where YY is the year (e.g., 22 -> 2022)
    accession_match = re.match(r'(\d+)-(\d{2})-(\d+)', filing_dir_name)
    if accession_match:
        year_short = int(accession_match.group(2))
        year = 2000 + year_short  # Convert to 4-digit year (22 -> 2022)
        # For annual filings, we can estimate quarter from year
        if filing_type in ["10-K", "10K"]:
            fiscal_quarter = None
            date_filed = f"{year}-09-30"  # Approximate fiscal year end for September filers
        else:
            fiscal_quarter = "Q4"  # Default for 10-Q
            date_filed = f"{year}-06-30"
        return year, fiscal_quarter, date_filed
    
    # Fallback: try to extract a date from the directory name
    date_match = re.search(r'(\d{4})(\d{2})(\d{2})', filing_dir_name)
    if date_match:
        year = int(date_match.group(1))
        month = int(date_match.group(2))
        date_filed = f"{year}-{date_match.group(2)}-{date_match.group(3)}"
        quarter_map = {1: "Q1", 2: "Q1", 3: "Q1",
                       4: "Q2", 5: "Q2", 6: "Q2",
                       7: "Q3", 8: "Q3", 9: "Q3",
                       10: "Q4", 11: "Q4", 12: "Q4"}
        quarter = quarter_map.get(month, "Q1")
        if filing_type in ["10-K", "10K"]:
            quarter = None
        return year, quarter, date_filed
    
    return 2024, None, "2024-01-01"


def run_pipeline():
    """Main pipeline entry point."""
    logger.info("=== Financial Research Copilot — Ingestion Pipeline ===")

    downloader = SECDownloader()
    parser = FilingParser()
    chunker = Chunker()

    # Step 1: Download
    logger.info("Step 1/3: Downloading SEC filings...")
    downloader.download_all()

    # Step 2: Find all downloaded files
    logger.info("Step 2/3: Locating downloaded filing documents...")
    filing_files = downloader.find_all_downloaded_files()

    if not filing_files:
        logger.error("No filing files found. Check data/raw/ directory.")
        return

    # Group by ticker for output
    ticker_chunks: dict = {t: [] for t in TARGET_TICKERS}

    # Step 3: Parse and chunk
    logger.info("Step 3/3: Parsing and chunking filings...")
    for file_info in tqdm(filing_files, desc="Processing filings"):
        path: Path = file_info["path"]
        ticker: str = file_info["ticker"]
        filing_type: str = file_info["filing_type"]

        # Infer fiscal info from directory name
        fiscal_year, fiscal_quarter, date_filed = estimate_fiscal_info(
            path.parent.name, filing_type
        )

        company = TICKER_TO_COMPANY.get(ticker, ticker)

        try:
            # Parse the document
            sections = parser.parse(path)
            if not sections:
                logger.warning(f"No content extracted from {path.name}")
                continue

            # Chunk it
            chunks = chunker.chunk_document(
                sections=sections,
                ticker=ticker,
                filing_type=filing_type,
                fiscal_year=fiscal_year,
                fiscal_quarter=fiscal_quarter,
                date_filed=date_filed,
                company=company,
                source_file=path.name,
            )

            if ticker in ticker_chunks:
                ticker_chunks[ticker].extend(chunks)
            
            logger.info(f"{ticker} {filing_type}: {len(sections)} sections → {len(chunks)} chunks")

        except Exception as e:
            logger.error(f"Failed to process {path}: {e}")
            continue

    # Step 4: Save to .jsonl files
    total_chunks = 0
    for ticker, chunks in ticker_chunks.items():
        if not chunks:
            continue
        out_path = DATA_PROCESSED_DIR / f"{ticker}_chunks.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        logger.success(f"Saved {len(chunks)} chunks → {out_path}")
        total_chunks += len(chunks)

    # Save a summary manifest
    summary = {
        "total_chunks": total_chunks,
        "tickers": {
            ticker: len(chunks) for ticker, chunks in ticker_chunks.items() if chunks
        },
        "chunk_files": [str(DATA_PROCESSED_DIR / f"{t}_chunks.jsonl") 
                        for t in TARGET_TICKERS if ticker_chunks.get(t)]
    }
    summary_path = DATA_PROCESSED_DIR / "ingestion_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.success(f"Pipeline complete. {total_chunks} total chunks saved.")
    logger.info(f"Summary: {summary_path}")
    return summary


if __name__ == "__main__":
    run_pipeline()