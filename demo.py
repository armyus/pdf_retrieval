"""
PDF Semantic Retrieval & RAG System — Complete Demo Showcase

This interactive script guides you through every feature of the system:
  1. PDF Discovery (V0)
  2. Text Extraction & Keyword Search (V1)
  3. Dense Semantic Search (V2)
  4. Advanced Retrieval with Deduplication & Grouping (V3)
  5. Grounded RAG Question Answering with Citations (V4)

Usage:
    python demo.py
    python demo.py --folder ./data
"""

import argparse
import sys
import time

from src.pdf_scanner import PDFScanner, print_results
from src.pdf_parser import PDFParser
from src.keyword_search import KeywordSearcher, print_search_results
from src.config import RetrievalConfig
from src.retriever import SemanticRetriever, print_semantic_results
from src.rag import RAGPipeline, print_rag_result


def print_banner(title: str):
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75 + "\n")


def run_demo(folder_path: str = "./data"):
    print_banner("PDF SEMANTIC RETRIEVAL & RAG SYSTEM — DEMO WALKTHROUGH")
    print(f"Target PDF folder: {folder_path}\n")

    # -------------------------------------------------------------
    # STAGE 1: PDF Discovery (V0)
    # -------------------------------------------------------------
    print(">>> STAGE 1: Discovering PDF files in folder...")
    try:
        scanner = PDFScanner(folder_path)
    except Exception as e:
        print(f"Error initializing scanner: {e}")
        return

    pdf_files = scanner.scan()
    print_results(pdf_files)

    if not pdf_files:
        print("Please place at least one PDF file into the folder to run the demo.")
        return

    # -------------------------------------------------------------
    # STAGE 2: Text Extraction & Keyword Search (V1)
    # -------------------------------------------------------------
    print_banner("STAGE 2: PDF Text Extraction & Keyword Search (V1)")
    print("Extracting text and metadata from all discovered PDFs...")
    parser = PDFParser()
    documents = parser.parse_many([pdf.full_path for pdf in pdf_files])

    extracted_count = sum(1 for d in documents if d.has_text)
    print(f"Successfully extracted text from {extracted_count}/{len(documents)} PDFs.")

    keyword_query = "cryptography"
    print(f"\nRunning exact keyword search for: '{keyword_query}'...")
    keyword_searcher = KeywordSearcher()
    kw_results = keyword_searcher.search(documents, keyword_query)
    # Show top 2 for demo brevity
    print_search_results(kw_results[:2], keyword_query)

    # -------------------------------------------------------------
    # STAGE 3: Semantic Search with Dense Embeddings (V2)
    # -------------------------------------------------------------
    print_banner("STAGE 3: Dense Semantic Search (V2)")
    print("Indexing text chunks into in-memory vector store (sentence-transformers)...")
    config = RetrievalConfig.default()
    retriever = SemanticRetriever(config=config)
    total_chunks = retriever.index_documents(documents)
    print(f"Generated embeddings and indexed {total_chunks} text chunks.\n")

    conceptual_query = "How do we keep private messages secret using prime numbers?"
    print(f"Asking a conceptual query (no exact keywords required):\n> \"{conceptual_query}\"")
    sem_results = retriever.search(conceptual_query, top_k=2)
    print_semantic_results(sem_results, conceptual_query)

    # -------------------------------------------------------------
    # STAGE 4: Better Retrieval with Deduplication & Grouping (V3)
    # -------------------------------------------------------------
    print_banner("STAGE 4: Refined Retrieval with Presets & Document Grouping (V3)")
    v3_query = "exploratory data analysis graphical techniques"
    print(f"Query: \"{v3_query}\" (Preset: Precise, Threshold: 0.35, Deduplication: ON)")

    v3_results = retriever.search(v3_query, top_k=3, score_threshold=0.35, deduplicate=True)
    print_semantic_results(v3_results, v3_query, group_by_doc=True)

    # -------------------------------------------------------------
    # STAGE 5: Grounded RAG Question Answering with Citations (V4)
    # -------------------------------------------------------------
    print_banner("STAGE 5: Grounded RAG (Retrieval-Augmented Generation) (V4)")
    rag_query = "What are the main graphical techniques used in exploratory data analysis?"
    print(f"Question: \"{rag_query}\"")
    print("Synthesizing grounded answer with numbered citations...\n")

    rag_pipeline = RAGPipeline(retriever=retriever, llm_backend="auto")
    rag_result = rag_pipeline.answer(rag_query, top_k=3)
    print_rag_result(rag_result)

    print("\n" + "=" * 75)
    print("  DEMO COMPLETED SUCCESSFULLY!")
    print("  You can now run custom queries using:")
    print("    python -m src.main --folder ./data --query \"Your Question\" --rag")
    print("    python -m src.main --folder ./data --search --rag (Interactive Mode)")
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="PDF Semantic Retrieval & RAG System Demo")
    parser.add_argument("--folder", type=str, default="./data", help="Path to PDF folder (default: ./data)")
    args = parser.parse_args()
    run_demo(args.folder)


if __name__ == "__main__":
    main()
