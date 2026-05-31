"""
Downloads SEC filings for target tickers using sec-edgar-downloader.
Organises downloads into data/raw/{ticker}/{filing_type}/
Tracks what has already been downloaded to avoid re-downloading.
"""

from sec_edgar_downloader import Downloader
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import json
import time
from config import DATA_RAW_DIR, TARGET_TICKERS, FILING_TYPES, FILINGS_PER_TYPE


class SECDownloader:
    def __init__(self):
        # sec-edgar-downloader requires a company name and email for the 
        # SEC EDGAR user-agent header (required by SEC's fair use policy)
        self.dl = Downloader(
            company_name="FinancialCopilot Research",
            email_address="research@financialcopilot.local",
            download_folder=str(DATA_RAW_DIR)
        )
        self.manifest_path = DATA_RAW_DIR / "download_manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        """Track downloaded filings to avoid re-downloading on re-runs."""
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text())
        return {}

    def _save_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2))

    def _manifest_key(self, ticker: str, filing_type: str) -> str:
        return f"{ticker}_{filing_type}"

    def download_ticker(self, ticker: str, filing_type: str) -> list:
        """
        Download filings for one ticker + filing type.
        Returns list of paths to downloaded filing folders.
        Skip if already in manifest.
        """
        key = self._manifest_key(ticker, filing_type)
        if key in self.manifest:
            logger.info(f"Skipping {ticker} {filing_type} — already downloaded")
            target_dir = DATA_RAW_DIR / "sec-edgar-filings" / ticker / filing_type.replace("-", "")
            if target_dir.exists():
                return list(target_dir.glob("*/"))
            return []

        logger.info(f"Downloading {FILINGS_PER_TYPE}x {filing_type} for {ticker}")
        try:
            self.dl.get(
                form=filing_type,
                ticker_or_cik=ticker,
                limit=FILINGS_PER_TYPE,
                download_details=True,
            )
            self.manifest[key] = {"status": "ok", "count": FILINGS_PER_TYPE}
            self._save_manifest()
            logger.success(f"Downloaded {filing_type} for {ticker}")
        except Exception as e:
            logger.error(f"Failed to download {filing_type} for {ticker}: {e}")
            self.manifest[key] = {"status": "error", "error": str(e)}
            self._save_manifest()
            return []

        # sec-edgar-downloader saves to: data/raw/sec-edgar-filings/{ticker}/{filing_type}/
        target_dir = DATA_RAW_DIR / "sec-edgar-filings" / ticker / filing_type.replace("-", "")
        if not target_dir.exists():
            # Try alternate path format
            target_dir = DATA_RAW_DIR / "sec-edgar-filings" / ticker / filing_type
        if target_dir.exists():
            return list(target_dir.glob("*/"))
        return []

    def download_all(self) -> dict:
        """
        Download all configured tickers and filing types.
        Returns dict: {ticker: [list of filing folder paths]}
        """
        results = {}
        total = len(TARGET_TICKERS) * len(FILING_TYPES)
        with tqdm(total=total, desc="Downloading filings") as pbar:
            for ticker in TARGET_TICKERS:
                results[ticker] = []
                for filing_type in FILING_TYPES:
                    paths = self.download_ticker(ticker, filing_type)
                    results[ticker].extend(paths)
                    pbar.update(1)
                    time.sleep(1)  # Rate limiting for SEC EDGAR
        return results

    def find_all_downloaded_files(self) -> list:
        """
        Walk data/raw/ and return metadata for every .htm/.html/.txt file found.
        Returns list of dicts with keys: path, ticker, filing_type
        Also finds any PDF files.
        """
        filing_root = DATA_RAW_DIR / "sec-edgar-filings"
        if not filing_root.exists():
            logger.warning(f"No filings found at {filing_root}")
            return []

        files = []
        for ticker_dir in filing_root.iterdir():
            if not ticker_dir.is_dir():
                continue
            ticker = ticker_dir.name
            for filing_type_dir in ticker_dir.iterdir():
                if not filing_type_dir.is_dir():
                    continue
                filing_type = filing_type_dir.name
                for filing_dir in filing_type_dir.iterdir():
                    if not filing_dir.is_dir():
                        continue
                    # Look for primary document (htm preferred, fallback to txt)
                    for ext in ["*.htm", "*.html", "*.txt", "*.pdf"]:
                        for f in filing_dir.glob(ext):
                            if "full-submission" not in f.name.lower():
                                files.append({
                                    "path": f,
                                    "ticker": ticker,
                                    "filing_type": filing_type,
                                    "filing_dir": str(filing_dir)
                                })
                            break  # one file per folder
        logger.info(f"Found {len(files)} filing documents")
        return files