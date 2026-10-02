"""
PDF Semantic Retrieval & RAG System — CLI Entry Point

Usage:
    # V0: PDF discovery only
    python -m src.main --folder ./data

    # V1: Keyword search
    python -m src.main --folder ./data --query "cryptography" --keyword

    # V2/V3: Semantic search with presets & thresholds
    python -m src.main --folder ./data --query "How do machines learn?" --semantic --preset precise

    # V4: Grounded RAG with Local LLM / Extractive synthesis & citations
    python -m src.main --folder ./data --query "How does RSA encryption work and what role do primes play?" --rag
    python -m src.main --folder ./data --search --rag
"""

import argparse
import sys

from src.pdf_scanner import PDFScanner, print_results
from src.config import RetrievalConfig


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PDF Semantic Retrieval & RAG System",
    )
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Path to the folder containing PDF files.",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="Enter interactive search/question mode after scanning.",
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Use RAG (Retrieval-Augmented Generation) to synthesize grounded answers with citations.",
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Use semantic search with embeddings (default when --rag is enabled).",
    )
    parser.add_argument(
        "--keyword",
        action="store_true",
        help="Explicitly use keyword search instead of semantic search / RAG.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a single search query or question and exit (non-interactive).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of results to retrieve (default: from config preset or 5).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Minimum similarity score threshold (default: from config preset or 0.30).",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["default", "precise", "broad"],
        default="default",
        help="Retrieval configuration preset (default, precise, broad).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override chunk size in characters.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Override chunk overlap in characters.",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable near-duplicate and same-page chunk deduplication.",
    )
    parser.add_argument(
        "--group-by-doc",
        action="store_true",
        help="Group retrieved passages under their parent document headings.",
    )
    parser.add_argument(
        "--filter-doc",
        type=str,
        default=None,
        help="Filter search results to documents matching this name substring.",
    )
    parser.add_argument(
        "--llm-backend",
        type=str,
        choices=["auto", "ollama", "llama_cpp", "extractive"],
        default="auto",
        help="LLM backend for RAG answer generation (auto, ollama, llama_cpp, extractive).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name for Ollama or file path for GGUF model.",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> RetrievalConfig:
    """Constructs a RetrievalConfig from preset and explicit CLI overrides."""
    if args.preset == "precise":
        config = RetrievalConfig.precise()
    elif args.preset == "broad":
        config = RetrievalConfig.broad()
    else:
        config = RetrievalConfig.default()

    if args.chunk_size is not None:
        config.chunk_size = args.chunk_size
    if args.chunk_overlap is not None:
        config.chunk_overlap = args.chunk_overlap
    if args.top_k is not None:
        config.top_k = args.top_k
    if args.threshold is not None:
        config.score_threshold = args.threshold
    if args.no_dedup:
        config.deduplicate = False

    return config


def main(argv=None) -> int:
    """Main entry point. Returns 0 on success, 1 on error."""
    args = parse_args(argv)

    print("=" * 50)
    print("  PDF Knowledge Retrieval & RAG System")
    print("=" * 50)

    # --- V0: Discovery ---
    try:
        scanner = PDFScanner(args.folder)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"\nError: {e}")
        return 1

    pdf_files = scanner.scan()
    print_results(pdf_files)

    if not pdf_files:
        return 0

    use_rag = args.rag
    use_semantic = args.semantic or use_rag
    is_interactive = args.search and not args.query

    # If neither query nor search mode requested, stop after V0 discovery
    if not args.search and not args.query:
        return 0

    # --- Parse PDFs ---
    from src.pdf_parser import PDFParser
    print("Extracting text from PDFs...")
    parser = PDFParser()
    documents = parser.parse_many([pdf.full_path for pdf in pdf_files])

    success = [d for d in documents if not d.error and d.has_text]
    no_text = [d for d in documents if not d.error and not d.has_text]
    errors = [d for d in documents if d.error]

    print(f"\nExtraction complete:")
    print(f"  {len(success)} PDFs with extractable text")
    if no_text:
        print(f"  {len(no_text)} PDFs with no extractable text (scanned/image PDFs)")
    if errors:
        print(f"  {len(errors)} PDFs failed to parse")
        for d in errors:
            print(f"    - {d.filename}: {d.error}")
    print()

    # Document filter helper
    doc_filter_fn = None
    if args.filter_doc:
        substr = args.filter_doc.lower()
        doc_filter_fn = lambda fn: substr in fn.lower()

    # --- V4: RAG Mode ---
    if use_rag:
        from src.retriever import SemanticRetriever
        from src.rag import RAGPipeline, print_rag_result

        config = build_config(args)
        print(f"Configuring RAG Pipeline ({args.preset} preset, backend: {args.llm_backend})...")
        print("\nBuilding semantic vector index (generating embeddings)...")
        retriever = SemanticRetriever(config=config)
        total_chunks = retriever.index_documents(documents)
        print(f"Indexed {total_chunks} text chunks into vector store.\n")

        rag_pipeline = RAGPipeline(
            retriever=retriever,
            llm_backend=args.llm_backend,
            model_name_or_path=args.model_name,
        )

        if args.query:
            result = rag_pipeline.answer(
                query=args.query,
                top_k=config.top_k,
                score_threshold=config.score_threshold,
                deduplicate=config.deduplicate,
            )
            print_rag_result(result)
            return 0

        if is_interactive:
            print("=== Interactive Grounded RAG Question Answering ===")
            print("Ask questions about your documents (or type 'quit' to exit):")
            while True:
                try:
                    query = input("\n> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break

                if query.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break

                if not query:
                    continue

                result = rag_pipeline.answer(
                    query=query,
                    top_k=config.top_k,
                    score_threshold=config.score_threshold,
                    deduplicate=config.deduplicate,
                )
                print_rag_result(result)

    # --- V2/V3: Semantic Search Mode ---
    elif use_semantic:
        from src.retriever import SemanticRetriever, print_semantic_results

        config = build_config(args)
        print(f"Retrieval configuration ({args.preset} preset):")
        print(f"  Chunk size: {config.chunk_size} | Overlap: {config.chunk_overlap}")
        print(f"  Top-K: {config.top_k} | Score Threshold: {config.score_threshold}")
        print(f"  Deduplication: {config.deduplicate}")
        print("\nBuilding semantic vector index (generating embeddings)...")

        retriever = SemanticRetriever(config=config)
        total_chunks = retriever.index_documents(documents)
        print(f"Indexed {total_chunks} text chunks into vector store.\n")

        if args.query:
            results = retriever.search(
                query=args.query,
                top_k=config.top_k,
                score_threshold=config.score_threshold,
                deduplicate=config.deduplicate,
                doc_filter=doc_filter_fn,
            )
            print_semantic_results(
                results,
                args.query,
                show_metadata=True,
                group_by_doc=args.group_by_doc,
            )
            return 0

        if is_interactive:
            print("=== Interactive Semantic Search Mode ===")
            print("Enter your question/concept (or 'quit' to exit):")
            while True:
                try:
                    query = input("\n> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break

                if query.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break

                if not query:
                    continue

                results = retriever.search(
                    query=query,
                    top_k=config.top_k,
                    score_threshold=config.score_threshold,
                    deduplicate=config.deduplicate,
                    doc_filter=doc_filter_fn,
                )
                print_semantic_results(
                    results,
                    query,
                    show_metadata=True,
                    group_by_doc=args.group_by_doc,
                )

    # --- V1: Keyword Search Mode ---
    else:
        from src.keyword_search import KeywordSearcher, print_search_results

        searcher = KeywordSearcher()
        search_docs = documents
        if doc_filter_fn:
            search_docs = [d for d in documents if doc_filter_fn(d.filename)]

        if args.query:
            results = searcher.search(search_docs, args.query)
            print_search_results(results, args.query)
            return 0

        if is_interactive:
            print("=== Interactive Keyword Search Mode ===")
            print("Enter your keyword query (or 'quit' to exit):")
            while True:
                try:
                    query = input("\n> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break

                if query.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break

                if not query:
                    continue

                results = searcher.search(search_docs, query)
                print_search_results(results, query)

    return 0


if __name__ == "__main__":
    sys.exit(main())
