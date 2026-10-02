"""
Chunker — V3: Enhanced Text Chunking Module

Splits page-level extracted PDF text into overlapping chunks
while preserving rich document, page, and chunk-level metadata.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import re

from src.pdf_parser import ParsedPDF, PageContent


@dataclass
class TextChunk:
    """Represents a discrete chunk of text with complete hierarchical metadata."""
    chunk_id: str                   # Unique identifier, e.g., "filename_p1_c0"
    filename: str
    full_path: str
    page_number: int                # 1-indexed
    text: str
    chunk_index: int = 0            # 0-indexed index of chunk within its page
    total_pages: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)  # Document/page metadata
    char_count: int = 0

    def __post_init__(self):
        self.char_count = len(self.text)


class TextChunker:
    """Splits document pages into text chunks with configurable size and overlap."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 75):
        """
        Args:
            chunk_size: Target size in characters per chunk.
            chunk_overlap: Overlap in characters between consecutive chunks.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> List[str]:
        """
        Splits a single string of text into overlapping chunks.
        Cleans redundant whitespace and handles boundaries cleanly.
        """
        text = text.strip()
        if not text:
            return []

        # Normalize consecutive whitespaces while preserving single newlines / spaces
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        step = self.chunk_size - self.chunk_overlap

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start += step

        return chunks

    def chunk_page(
        self,
        page: PageContent,
        filename: str,
        full_path: str,
        total_pages: int = 1,
        doc_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[TextChunk]:
        """
        Chunks a single PageContent object into TextChunks with metadata.
        """
        if not page.has_text:
            return []

        raw_chunks = self.chunk_text(page.text)
        text_chunks: List[TextChunk] = []
        meta = dict(doc_metadata or {})

        for idx, chunk_str in enumerate(raw_chunks):
            chunk_id = f"{filename}_p{page.page_number}_c{idx}"
            text_chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    filename=filename,
                    full_path=full_path,
                    page_number=page.page_number,
                    chunk_index=idx,
                    total_pages=total_pages,
                    text=chunk_str,
                    metadata=meta,
                )
            )

        return text_chunks

    def chunk_document(self, document: ParsedPDF) -> List[TextChunk]:
        """
        Chunks an entire ParsedPDF across all pages preserving doc metadata.
        """
        if document.error or not document.has_text:
            return []

        chunks: List[TextChunk] = []
        for page in document.pages_with_text:
            chunks.extend(
                self.chunk_page(
                    page=page,
                    filename=document.filename,
                    full_path=document.full_path,
                    total_pages=document.total_pages,
                    doc_metadata=document.metadata,
                )
            )

        return chunks

    def chunk_documents(self, documents: List[ParsedPDF]) -> List[TextChunk]:
        """
        Chunks a collection of ParsedPDF documents.
        """
        all_chunks: List[TextChunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
