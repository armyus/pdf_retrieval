# PDF Semantic Retrieval & RAG System

A modular, local-first **Personal PDF Knowledge Retrieval & Question-Answering System**.

Point it at a folder which filled with arbitrary PDFs and ask questions in natural language. The system retrieves the most relevant passages and generates grounded answers citing specific documents and page numbers — without requiring you to remember filenames, paper titles, or exact keyword matches.

---

## 🌟 Interactive Web Dashboard

Launch the browser-based dashboard with a single command:

```bash
python app.py --folder ./data
```
Open your browser at **http://localhost:8000** to explore:
- 💬 **Grounded RAG Assistant**: Ask questions and get answers with interactive citation cards.
- 🔍 **Multi-Strategy Search**: Switch between Hybrid, Dense Vector, BM25, and Exact Match.
- ⚖️ **Strategy Comparison**: Compare Dense vs BM25 results side-by-side for the same query.
- 📊 **Evaluation Benchmark**: Run information retrieval benchmarks (P@K, MRR, MAP, nDCG) live.
- 📁 **Document Repository Explorer**: Inspect discovered PDFs, pages, and extracted statistics.

---

## Table of Contents

1. [Features & Capabilities](#features--capabilities)
2. [Prerequisites & Installation](#prerequisites--installation)
3. [Interactive Web Dashboard](#interactive-web-dashboard)
4. [One-Command Terminal Demo](#one-command-terminal-demo)
5. [User Guide: How to Use This Technology](#user-guide-how-to-use-this-technology)
   - [Mode 1: Web Dashboard](#1-launch-the-web-dashboard-recommended)
   - [Mode 2: Discovering PDFs](#2-discovering-pdfs-v0)
   - [Mode 3: Exact Keyword Search](#3-exact-keyword-search-v1)
   - [Mode 4: Semantic Concept Search](#4-semantic-concept-search-v2)
   - [Mode 5: Advanced Filtered & Grouped Search](#5-advanced-retrieval-presets--grouping-v3)
   - [Mode 6: Grounded RAG Question Answering](#6-grounded-rag-question-answering-v4)
   - [Mode 7: Advanced Hybrid Retrieval (Dense + BM25)](#7-advanced-hybrid-retrieval-v5)
   - [Mode 8: Quantitative Evaluation Benchmark](#8-quantitative-evaluation-benchmark-v6)
   - [Mode 9: Interactive Terminal Mode](#9-interactive-terminal-mode)
6. [Swappable Retrievers Architecture](#swappable-retrievers-architecture)
7. [Configuring LLM Backends](#configuring-llm-backends)
8. [Running Tests](#running-tests)
9. [Project Structure](#project-structure)
10. [Roadmap](#roadmap)

---

## Features & Capabilities

- **V0 — PDF Discovery**: Recursively scans directories, detects `.pdf` files (case-insensitive), reports names, paths, and file sizes.
- **V1 — Text Extraction & Keyword Search**: Page-level extraction using PyMuPDF with match count ranking and context snippets.
- **V2 — Semantic Dense Search**: Sliding-window chunking, dense vector embeddings with `sentence-transformers` (`all-MiniLM-L6-v2`), and in-memory vector store indexing.
- **V3 — Better Retrieval**:
  - Configuration presets: `default`, `precise`, `broad` (`RetrievalConfig`).
  - Full hierarchical tracking: `Document` $\rightarrow$ `Page` $\rightarrow$ `Chunk` $\rightarrow$ `Score`.
  - Token Jaccard similarity & adjacent same-page chunk deduplication (`ResultPostprocessor`).
  - Similarity score cutoff thresholding & document filtering.
  - Grouped document presentation (`--group-by-doc`).
- **V4 — Grounded RAG**:
  - Anti-hallucination prompt construction with numbered citation mapping (`[1]`, `[2]`).
  - Modular LLM backends: `Ollama` REST API, `llama-cpp-python` GGUF, and zero-dependency offline `ExtractiveFallback` synthesizer.
  - Insufficient context guardrails (refuses to invent facts when evidence is missing).
- **V5 — Advanced Retrieval (Hybrid & Swappable Retrievers)**:
  - **Pure Python BM25Okapi Index**: Fast sparse keyword search with IDF precomputation & term saturation.
  - **Hybrid Fusion**: Combines dense vector semantics with sparse BM25 keyword rankings via **Reciprocal Rank Fusion (RRF)** or **Weighted Score Normalization**.
  - **Swappable Architecture**: Easily switch between `hybrid`, `dense`, `bm25`, and `keyword` strategies.
- **V6 — Evaluation Framework & Interactive Web Dashboard**:
  - **Information Retrieval Benchmark Suite**: Precision@K, Recall@K, Mean Reciprocal Rank (MRR), Mean Average Precision (MAP), and nDCG@K.
  - **Flask Web Dashboard**: Responsive user interface for visual search, grounded RAG, side-by-side strategy comparison, and live benchmark evaluation.

---

## Prerequisites & Installation

### Requirements
- **Python 3.10+** (Tested on Python 3.11)
- Windows, macOS, or Linux

### Installation

1. **Clone or navigate to the repository:**
   ```bash
   cd pdf_retrieval
   ```

2. **(Optional) Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux / macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Interactive Web Dashboard

To launch the web dashboard:

```bash
# Shortcut launcher:
python app.py --folder ./data --port 8000

# Or via main module:
python -m src.main --folder ./data --dashboard --port 8000
```

Open **http://localhost:8000** in your browser.

---

## One-Command Terminal Demo

To see every terminal capability running in sequence across sample PDFs:

```bash
python demo.py --folder ./data
```

This runs:
1. PDF discovery
2. Text extraction & keyword search
3. Semantic dense vector search
4. Deduplicated and document-grouped retrieval
5. Grounded RAG answer synthesis with numbered citations
6. Advanced Hybrid Retrieval (Dense + BM25 with Reciprocal Rank Fusion)

---

## User Guide: How to Use This Technology

### 1. Launch the Web Dashboard (Recommended)
```bash
python app.py --folder ./data
```

### 2. Discovering PDFs (V0)
Scan a folder and list all discovered PDF files with sizes:
```bash
python -m src.main --folder ./data
```

### 3. Exact Keyword Search (V1)
Find all occurrences of a specific phrase across all PDFs:
```bash
python -m src.main --folder ./data --query "cryptography" --strategy keyword
```

### 4. Semantic Concept Search (V2)
Search conceptually without needing exact keyword matches:
```bash
python -m src.main --folder ./data --query "How do we keep private messages secret with prime numbers?" --strategy dense
```

### 5. Advanced Retrieval: Presets & Grouping (V3)
Use precision presets, score thresholds, and group results by parent document:
```bash
# High-precision preset with document grouping
python -m src.main --folder ./data --query "exploratory data analysis graphical techniques" --strategy dense --preset precise --group-by-doc

# Filter search to a specific document name
python -m src.main --folder ./data --query "clustering" --strategy dense --threshold 0.35 --filter-doc "ANALYSIS"
```

### 6. Grounded RAG Question Answering (V4)
Ask a natural-language question and get a synthesized answer backed by citations:
```bash
python -m src.main --folder ./data --query "What are the main graphical techniques used in exploratory data analysis?" --rag
```

### 7. Advanced Hybrid Retrieval (V5)
Combine dense semantic vector similarity with sparse BM25 keyword matching via Reciprocal Rank Fusion:
```bash
# Hybrid search across all PDFs
python -m src.main --folder ./data --query "cryptographic public key RSA prime factors" --strategy hybrid --top-k 3

# Grounded RAG powered by Hybrid Retrieval
python -m src.main --folder ./data --query "How does RSA encryption ensure message confidentiality?" --rag --strategy hybrid
```

### 8. Quantitative Evaluation Benchmark (V6)
Run information retrieval metrics across retrieval strategies:
```bash
python -m src.main --folder ./data --evaluate
```

**Leaderboard Output:**
```text
================================================================================
  RETRIEVAL BENCHMARK LEADERBOARD (V6 Evaluation)
================================================================================
Strategy     | P@1      | P@3      | P@5      | MRR      | MAP      | nDCG@5  
--------------------------------------------------------------------------------
BM25         | 1.0000   | 0.9333   | 0.8800   | 1.0000   | 2.2000   | 1.0000  
DENSE        | 1.0000   | 0.8000   | 0.8800   | 1.0000   | 2.0650   | 0.9618  
HYBRID       | 1.0000   | 0.8667   | 0.8400   | 1.0000   | 2.1000   | 1.0000  
================================================================================
```

### 9. Interactive Terminal Mode
Start an interactive conversational session over your PDFs:
```bash
# Interactive Hybrid RAG QA (Recommended):
python -m src.main --folder ./data --search --rag --strategy hybrid

# Interactive Semantic Search:
python -m src.main --folder ./data --search --strategy dense
```

---

## Swappable Retrievers Architecture

```text
BaseRetriever (Interface)
├── DenseRetriever       (sentence-transformers + InMemoryVectorStore)
├── BM25Retriever        (Pure-Python BM25Okapi Index)
├── HybridRetriever      (Fuses Dense + BM25 via Reciprocal Rank Fusion)
└── KeywordRetriever     (Exact case-insensitive substring search)
```

Factory usage:
```python
from src.retrievers import get_retriever
from src.config import RetrievalConfig

config = RetrievalConfig.default()
retriever = get_retriever("hybrid", config)
retriever.index_documents(documents)
results = retriever.search("your query", top_k=5)
```

---

## Configuring LLM Backends

The system supports multiple execution backends via `--llm-backend`:

1. **Auto (Default)**: Checks if local Ollama is active; otherwise uses the zero-dependency offline Extractive Synthesizer.
2. **Ollama (`--llm-backend ollama`)**:
   - Ensure Ollama is running (`ollama run llama3.2` or `ollama serve`).
   - Run: `python -m src.main --folder ./data --query "..." --rag --llm-backend ollama --model-name llama3.2`
3. **Local GGUF (`--llm-backend llama_cpp`)**:
   - Install `llama-cpp-python` (`pip install llama-cpp-python`).
   - Run: `python -m src.main --folder ./data --query "..." --rag --llm-backend llama_cpp --model-name ./models/model.gguf`
4. **Offline Extractive Synthesizer (`--llm-backend extractive`)**:
   - Works immediately on any CPU with zero external services or model downloads.

---

## Running Tests

The test suite covers every component with 100% pass rate:

```bash
python -m pytest tests/ -v
```

**126 tests** passing:
- 23 PDF scanner tests
- 22 PDF parser & metadata extraction tests
- 16 Keyword search tests
- 9 Hierarchical chunker tests
- 4 Embedding generator tests
- 5 Vector store tests
- 4 Semantic retriever tests
- 5 Configuration preset tests
- 5 Postprocessor / Deduplication tests
- 3 Grounded prompt builder tests
- 5 LLM engine tests
- 4 RAG pipeline tests
- 4 BM25 engine tests
- 5 Swappable retrievers & hybrid fusion tests
- 7 Evaluation framework tests
- 4 Web Dashboard API tests

---

## Project Structure

```text
pdf_retrieval/
├── data/                          # Target directory for user PDF documents
│   ├── content.pdf
│   └── ...
├── src/
│   ├── __init__.py
│   ├── main.py                    # Unified CLI entry point
│   ├── config.py                  # V3/V5: Configuration presets & parameters
│   ├── postprocessing.py          # V3: Deduplication, Jaccard suppression, filtering
│   ├── pdf_scanner.py             # V0: PDF discovery module
│   ├── pdf_parser.py              # V1: Text & document metadata extraction
│   ├── keyword_search.py          # V1: Keyword search engine
│   ├── chunker.py                 # V2/V3: Hierarchical text chunker
│   ├── embeddings.py              # V2: sentence-transformers embedding generator
│   ├── vector_store.py            # V2: In-memory vector store & cosine scoring
│   ├── bm25.py                    # V5: Pure Python BM25Okapi index & scoring
│   ├── retrievers.py              # V5: Swappable retrievers (Hybrid, Dense, BM25, Keyword)
│   ├── retriever.py               # Unified retriever exports & display helpers
│   ├── prompt_builder.py          # V4: Grounded context prompt builder
│   ├── llm_engine.py              # V4: Modular LLM backends (Ollama, GGUF, Extractive)
│   ├── rag.py                     # V4: End-to-end RAG question answering pipeline
│   ├── evaluation.py              # V6: Information Retrieval evaluation benchmark suite
│   └── dashboard.py               # V6: Interactive Flask Web Dashboard & REST API
├── tests/
│   ├── __init__.py
│   ├── test_pdf_scanner.py        # 23 tests
│   ├── test_pdf_parser.py         # 22 tests
│   ├── test_keyword_search.py     # 16 tests
│   ├── test_chunker.py            # 9 tests
│   ├── test_embeddings.py         # 4 tests
│   ├── test_vector_store.py       # 5 tests
│   ├── test_retriever.py          # 4 tests
│   ├── test_config.py             # 5 tests
│   ├── test_postprocessing.py     # 5 tests
│   ├── test_prompt_builder.py     # 3 tests
│   ├── test_llm_engine.py         # 5 tests
│   ├── test_rag.py                # 4 tests
│   ├── test_bm25.py               # 4 tests
│   ├── test_retrievers.py         # 5 tests
│   ├── test_evaluation.py         # 7 tests
│   └── test_dashboard.py          # 4 tests (Total: 126 tests)
├── app.py                         # Web Dashboard root launcher
├── demo.py                        # Interactive showcase demo script
├── requirements.txt               # Project dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # Complete system documentation
```

---

## Roadmap

| Version | Feature                          | Status  |
|---------|----------------------------------|---------|
| V0      | PDF Discovery                    | ✅ Done  |
| V1      | Text Extraction + Keyword Search | ✅ Done  |
| V2      | Semantic Search (Embeddings)     | ✅ Done  |
| V3      | Better Retrieval                 | ✅ Done  |
| V4      | RAG (LLM-powered answers)        | ✅ Done  |
| V5      | Advanced Retrieval (Hybrid/BM25) | ✅ Done  |
| V6      | Evaluation & Web Dashboard       | ✅ Done  |

---

## License

Personal / educational project.
