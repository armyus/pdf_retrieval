"""
PDF Semantic Retrieval & RAG System — Web Dashboard Launcher

Launches the interactive Web UI on http://localhost:8000.

Usage:
    python app.py
    python app.py --folder ./data --port 8000
"""

import sys
from src.dashboard import main

if __name__ == "__main__":
    main()
