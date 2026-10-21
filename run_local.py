"""One-command local setup and launch: python run_local.py

Creates .venv, installs dependencies, downloads the spaCy model, creates .env if missing,
then starts the app and opens the browser. Safe to re-run; finished steps are skipped.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
import venv
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
BIN = VENV / ("Scripts" if os.name == "nt" else "bin")
PY = BIN / ("python.exe" if os.name == "nt" else "python")
STAMP = VENV / ".deps-installed"
HOST, PORT = "127.0.0.1", int(os.environ.get("PORT", "8000"))


def run(*cmd: str) -> None:
    subprocess.run([str(c) for c in cmd], cwd=ROOT, check=True)


def ensure_python() -> None:
    if sys.version_info[:2] < (3, 11):  # noqa: UP036 - friendlier than a pip failure
        sys.exit("Python 3.11+ is required.")


def ensure_venv() -> None:
    if not PY.exists():
        print("Creating virtual environment (.venv)...")
        venv.create(VENV, with_pip=True)


def ensure_deps() -> None:
    reqs = ROOT / "requirements.txt"
    if STAMP.exists() and STAMP.stat().st_mtime >= reqs.stat().st_mtime:
        return
    print("Installing dependencies (first run takes a few minutes)...")
    run(PY, "-m", "pip", "install", "--quiet", "--upgrade", "pip")
    run(PY, "-m", "pip", "install", "-r", "requirements.txt")
    STAMP.touch()


def ensure_spacy_model() -> None:
    check = subprocess.run([str(PY), "-c", "import spacy; spacy.load('en_core_web_sm')"],
                           cwd=ROOT, capture_output=True)
    if check.returncode != 0:
        print("Downloading the spaCy English model...")
        run(PY, "-m", "spacy", "download", "en_core_web_sm")


def ensure_env() -> None:
    env, example = ROOT / ".env", ROOT / ".env.example"
    if not env.exists():
        # Default to no LLM so the app runs with zero accounts or keys.
        text = example.read_text(encoding="utf-8").replace("LLM_PROVIDER=groq", "LLM_PROVIDER=none")
        env.write_text(text, encoding="utf-8")
        print("Created .env with LLM_PROVIDER=none. Add a Gemini or Groq key there to enable the LLM.")


def warn_tesseract() -> None:
    windows_paths = (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                     r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe")
    if shutil.which("tesseract") or any(os.path.isfile(p) for p in windows_paths):
        return
    print("\nNote: Tesseract OCR was not found. Digital PDFs and DOCX still work, but scans and images won't.")
    print("  Windows: https://github.com/UB-Mannheim/tesseract/wiki")
    print("  macOS:   brew install tesseract")
    print("  Linux:   sudo apt install tesseract-ocr\n")


def open_browser_when_ready() -> None:
    import socket
    for _ in range(120):
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex((HOST, PORT)) == 0:
                webbrowser.open(f"http://{HOST}:{PORT}")
                return
        time.sleep(1)


def main() -> None:
    ensure_python()
    ensure_venv()
    ensure_deps()
    ensure_spacy_model()
    ensure_env()
    warn_tesseract()
    print(f"\nStarting DocIntel at http://{HOST}:{PORT}  (Ctrl+C to stop)")
    print("The first upload downloads the ~90 MB embedding model.\n")
    if "--no-browser" not in sys.argv:
        threading.Thread(target=open_browser_when_ready, daemon=True).start()
    try:
        run(PY, "-m", "uvicorn", "main:app", "--host", HOST, "--port", str(PORT))
    except (KeyboardInterrupt, subprocess.CalledProcessError):
        pass


if __name__ == "__main__":
    main()
