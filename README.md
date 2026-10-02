# PDF Semantic Retrieval & RAG System

A modular, local-first **Personal PDF Knowledge Retrieval & Question-Answering System**.

Point it at a folder filled with arbitrary PDFs and ask questions in natural language. The system retrieves the most relevant passages and generates grounded answers citing specific documents and page numbers — without requiring you to remember filenames, paper titles, or exact keyword matches.

---

## Table of Contents

1. [Features & Capabilities](#features--capabilities)
2. [Prerequisites & Installation](#prerequisites--installation)
3. [One-Command Demo](#one-command-demo)
4. [User Guide: How to Use This Technology](#user-guide-how-to-use-this-technology)
   - [Mode 1: Discovering PDFs](#1-discovering-pdfs-v0)
   - [Mode 2: Keyword Search](#2-exact-keyword-search-v1)
   - [Mode 3: Semantic Concept Search](#3-semantic-concept-search-v2)
   - [Mode 4: Advanced Filtered & Grouped Search](#4-advanced-retrieval-presets--grouping-v3)
   - [Mode 5: Grounded RAG Question Answering](#5-grounded-rag-question-answering-v4)
   - [Mode 6: Advanced Hybrid Retrieval (Dense + BM25)](#6-advanced-hybrid-retrieval-v5)
   - [Mode 7: Interactive Terminal QA Mode](#7-interactive-terminal-mode)
5. [Swappable Retrievers Architecture](#swappable-retrievers-architecture)
6. [Configuring LLM Backends](#configuring-llm-backends)
7. [Running Tests](#running-tests)
8. [Project Structure](#project-structure)
9. [Roadmap](#roadmap)

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
  - **Swappable Architecture**: Easily switch between `hybrid`, `dense`, `bm25`, and `keyword` strategies without code refactoring.

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

## One-Command Demo

Run the automated walkthrough that demonstrates every stage on sample PDFs:

```bash
python demo.py --folder ./data
```

This will showcase:
1. PDF discovery
2. Text extraction & keyword search
3. Semantic dense vector search
4. Deduplicated and document-grouped retrieval
5. Grounded RAG answer synthesis with numbered citations
6. Advanced Hybrid Retrieval (Dense + BM25 with Reciprocal Rank Fusion)

---

## User Guide: How to Use This Technology

Place your PDF documents in a folder (e.g. `./data` or `C:/path/to/my_papers/`) and choose any of the following modes:

### 1. Discovering PDFs (V0)
Scan a folder and list all discovered PDF files with sizes:
```bash
python -m src.main --folder ./data
```

### 2. Exact Keyword Search (V1)
Find all occurrences of a specific phrase across all PDFs:
```bash
python -m src.main --folder ./data --query "cryptography" --strategy keyword
```

### 3. Semantic Concept Search (V2)
Search conceptually without needing exact keyword matches:
```bash
python -m src.main --folder ./data --query "How do we keep private messages secret with prime numbers?" --strategy dense
```

### 4. Advanced Retrieval: Presets & Grouping (V3)
Use precision presets, score thresholds, and group results by parent document:
```bash
# High-precision preset with document grouping
python -m src.main --folder ./data --query "exploratory data analysis graphical techniques" --strategy dense --preset precise --group-by-doc

# Filter search to a specific document name
python -m src.main --folder ./data --query "clustering" --strategy dense --threshold 0.35 --filter-doc "ANALYSIS"
```

### 5. Grounded RAG Question Answering (V4)
Ask a natural-language question and get a synthesized answer backed by citations:
```bash
python -m src.main --folder ./data --query "What are the main graphical techniques used in exploratory data analysis?" --rag
```

### 6. Advanced Hybrid Retrieval (V5)
Combine dense semantic vector similarity with sparse BM25 keyword matching via Reciprocal Rank Fusion:
```bash
# Hybrid search across all PDFs
python -m src.main --folder ./data --query "cryptographic public key RSA prime factors" --strategy hybrid --top-k 3

# Grounded RAG powered by Hybrid Retrieval
python -m src.main --folder ./data --query "How does RSA encryption ensure message confidentiality?" --rag --strategy hybrid
```

### 7. Interactive Terminal Mode
Start an interactive conversational session over your PDFs:
```bash
# Interactive Hybrid RAG QA (Recommended):
python -m src.main --folder ./data --search --rag --strategy hybrid

# Interactive Semantic Search:
python -m src.main --folder ./data --search --strategy dense

# Interactive BM25 Search:
python -m src.main --folder ./data --search --strategy bm25
```

---

## Swappable Retrievers Architecture

The system provides a unified abstraction hierarchy allowing any retrieval strategy to be swapped dynamically:

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

**115 tests** passing:
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
│   └── rag.py                     # V4: End-to-end RAG question answering pipeline
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
│   └── test_retrievers.py         # 5 tests (Total: 115 tests)
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
| V6      | Evaluation Framework             | ⬜ Next  |

---

## License

Personal / educational project.
