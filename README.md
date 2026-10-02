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
   - [Mode 6: Interactive Terminal QA Mode](#6-interactive-terminal-mode)
5. [Configuring LLM Backends](#configuring-llm-backends)
6. [Architecture & Pipeline](#architecture--pipeline)
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
  - Grouped document presentation.
- **V4 — Grounded RAG**:
  - Anti-hallucination prompt construction with numbered citation mapping (`[1]`, `[2]`).
  - Modular LLM backends: `Ollama` REST API, `llama-cpp-python` GGUF, and zero-dependency offline `ExtractiveFallback` synthesizer.
  - Insufficient context guardrails (refuses to invent facts when evidence is missing).

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

This will run:
1. PDF discovery
2. Text extraction & keyword search
3. Semantic dense vector search
4. Deduplicated and document-grouped retrieval
5. Grounded RAG answer synthesis with numbered citations

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
python -m src.main --folder ./data --query "cryptography" --keyword
```

### 3. Semantic Concept Search (V2)
Search conceptually without needing exact keyword matches:
```bash
python -m src.main --folder ./data --query "How do we keep private messages secret with prime numbers?" --semantic
```

### 4. Advanced Retrieval: Presets & Grouping (V3)
Use precision presets, score thresholds, and group results by parent document:
```bash
# High-precision preset with document grouping
python -m src.main --folder ./data --query "exploratory data analysis graphical techniques" --semantic --preset precise --group-by-doc

# Filter search to a specific document name
python -m src.main --folder ./data --query "clustering" --semantic --threshold 0.35 --filter-doc "ANALYSIS"
```

### 5. Grounded RAG Question Answering (V4)
Ask a natural-language question and get a synthesized answer backed by citations:
```bash
python -m src.main --folder ./data --query "What are the main graphical techniques used in exploratory data analysis?" --rag
```

**Example Output:**
```text
===========================================================================
Question: What are the main graphical techniques used in exploratory data analysis?
===========================================================================

Answer:
Graphics play a central role in Exploratory Data Analysis (EDA) by enabling
analysts to visually examine data and quickly understand its underlying structure. [1]
Graphical representations are a fundamental part of Exploratory Data Analysis (EDA). [2]
EDA is an initial, crucial step in data analysis to understand datasets by
summarizing main characteristics, finding patterns, spotting anomalies, and checking
assumptions, often using graphs and visualizations. [3]

Sources & Citations:
  [1] EXPLORATORY DATA ANALYSIS- LECTURE NOTES.pdf — Page 5/16 (Score: 0.6548)
      Excerpt: "ROLE OF GRAPHICS IN DATA EXPLORATION: Graphics play a central role..."

  [2] EXPLORATORY DATA ANALYSIS- LECTURE NOTES.pdf — Page 6/16 (Score: 0.6528)
      Excerpt: "INTRODUCTION TO SINGLE, BI AND MULTIDIMENSIONAL GRAPHICAL REPRESENTATION..."
---------------------------------------------------------------------------
```

### 6. Interactive Terminal Mode
Start an interactive conversational session over your PDFs:
```bash
# Interactive RAG QA:
python -m src.main --folder ./data --search --rag

# Interactive Semantic Search:
python -m src.main --folder ./data --search --semantic

# Interactive Keyword Search:
python -m src.main --folder ./data --search --keyword
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

## Architecture & Pipeline

```text
User Question
      ↓
Query Processing
      ↓
Semantic / Dense Retriever (sentence-transformers + NumPy Vector Index)
      ↓
Result Postprocessing (Deduplication, Token Jaccard, Thresholding)
      ↓
Context Construction (PromptBuilder with Numbered [1], [2] References)
      ↓
Local LLM / Synthesizer Engine (Ollama / GGUF / Extractive)
      ↓
Grounded Answer + Structured Source Citations
```

---

## Running Tests

The test suite covers every component from unit logic to integration on real PDFs:

```bash
python -m pytest tests/ -v
```

**104 tests** passing with 100% success rate.

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
│   ├── config.py                  # V3: Configuration presets & parameters
│   ├── postprocessing.py          # V3: Deduplication, Jaccard suppression, filtering
│   ├── pdf_scanner.py             # V0: PDF discovery module
│   ├── pdf_parser.py              # V1: Text & document metadata extraction
│   ├── keyword_search.py          # V1: Keyword search engine
│   ├── chunker.py                 # V2/V3: Hierarchical text chunker
│   ├── embeddings.py              # V2: sentence-transformers embedding generator
│   ├── vector_store.py            # V2: In-memory vector store & cosine scoring
│   ├── retriever.py               # V2/V3: Semantic retrieval pipeline
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
│   ├── test_config.py             # 3 tests
│   ├── test_postprocessing.py     # 5 tests
│   ├── test_prompt_builder.py     # 3 tests
│   ├── test_llm_engine.py         # 5 tests
│   └── test_rag.py                # 4 tests (Total: 104 tests)
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
| V5      | Advanced Retrieval (Hybrid/BM25) | ⬜ Next  |
| V6      | Evaluation Framework             | ⬜       |

---

## License

Personal / educational project.
