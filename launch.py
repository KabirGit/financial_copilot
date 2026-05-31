"""
launch.py — waits for indexer to finish, then starts API and UI.
Run: python launch.py
"""

import subprocess
import sys
import time
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent / "data" / "processed"
BM25_INDEX = PROCESSED_DIR / "bm25_index.pkl"
POLL_INTERVAL = 30  # seconds between checks


def indexer_process_running() -> bool:
    result = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"],
        capture_output=True, text=True
    )
    return "rag.indexer" in result.stdout or "rag\\indexer" in result.stdout


def wait_for_indexer():
    print("[launch] Waiting for indexer to complete...")
    while True:
        if BM25_INDEX.exists():
            # Give it a few extra seconds to finish writing
            time.sleep(5)
            if BM25_INDEX.exists():
                print(f"[launch] BM25 index found at {BM25_INDEX}. Indexing complete.")
                return
        print(f"[launch] Index not ready yet, checking again in {POLL_INTERVAL}s...")
        time.sleep(POLL_INTERVAL)


def start_api():
    print("[launch] Starting API server on port 8080...")
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"],
        cwd=Path(__file__).parent,
    )


def start_ui():
    print("[launch] Starting Streamlit UI on port 8501...")
    return subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "ui/app.py",
         "--server.port", "8501", "--server.address", "0.0.0.0"],
        cwd=Path(__file__).parent,
    )


if __name__ == "__main__":
    wait_for_indexer()

    api_proc = start_api()
    time.sleep(5)  # let API bind before UI starts
    ui_proc = start_ui()

    print("\n[launch] All services running:")
    print("  API : http://localhost:8080")
    print("  Docs: http://localhost:8080/docs")
    print("  UI  : http://localhost:8501")
    print("\n[launch] Press Ctrl+C to stop all services.\n")

    try:
        api_proc.wait()
    except KeyboardInterrupt:
        print("\n[launch] Shutting down...")
        api_proc.terminate()
        ui_proc.terminate()
