"""
PDF Parser — V1: Text Extraction Module

Extracts text from PDF files page-by-page, preserving metadata
(filename, full path, page number, extracted text).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pymupdf  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    page_number: int        # 1-indexed
    text: str
    char_count: int = 0

    def __post_init__(self):
        self.char_count = len(self.text)

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())


@dataclass
class ParsedPDF:
    """Represents a fully parsed PDF document with all its page contents."""
    filename: str
    full_path: str
    pages: List[PageContent] = field(default_factory=list)
    total_pages: int = 0
    metadata: dict = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def has_text(self) -> bool:
        """True if at least one page has extractable text."""
        return any(page.has_text for page in self.pages)

    @property
    def full_text(self) -> str:
        """Concatenation of all page texts."""
        return "\n\n".join(page.text for page in self.pages if page.has_text)

    @property
    def pages_with_text(self) -> List[PageContent]:
        """Only pages that contain extractable text."""
        return [p for p in self.pages if p.has_text]


class PDFParser:
    """Extracts text from PDF files using PyMuPDF."""

    def parse(self, pdf_path: str) -> ParsedPDF:
        """
        Parse a single PDF file and extract text from all pages.

        Args:
            pdf_path: Absolute or relative path to the PDF file.

        Returns:
            A ParsedPDF object containing page-level extracted text,
            or an object with an error message if parsing failed.
        """
        path = Path(pdf_path).resolve()
        filename = path.name

        if not path.exists():
            return ParsedPDF(
                filename=filename,
                full_path=str(path),
                error=f"File not found: {path}",
            )

        if not path.suffix.lower() == ".pdf":
            return ParsedPDF(
                filename=filename,
                full_path=str(path),
                error=f"Not a PDF file: {path}",
            )

        try:
            doc = pymupdf.open(str(path))
        except Exception as e:
            return ParsedPDF(
                filename=filename,
                full_path=str(path),
                error=f"Failed to open PDF: {e}",
            )

        doc_metadata = {}
        try:
            doc_metadata = doc.metadata or {}
        except Exception:
            pass

        pages: List[PageContent] = []
        try:
            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    text = page.get_text("text")
                    pages.append(PageContent(
                        page_number=page_num + 1,  # 1-indexed
                        text=text,
                    ))
                except Exception as e:
                    logger.warning(
                        "Failed to extract text from page %d of %s: %s",
                        page_num + 1, filename, e,
                    )
                    pages.append(PageContent(
                        page_number=page_num + 1,
                        text="",
                    ))
        finally:
            doc.close()

        parsed = ParsedPDF(
            filename=filename,
            full_path=str(path),
            pages=pages,
            total_pages=len(pages),
            metadata=doc_metadata,
        )

        if not parsed.has_text:
            logger.warning("No extractable text in %s (scanned/image PDF?)", filename)

        return parsed

    def parse_many(self, pdf_paths: List[str]) -> List[ParsedPDF]:
        """
        Parse multiple PDF files.

        Args:
            pdf_paths: List of paths to PDF files.

        Returns:
            List of ParsedPDF objects (one per file), in the same order.
        """
        results: List[ParsedPDF] = []
        for i, path in enumerate(pdf_paths, start=1):
            filename = Path(path).name
            logger.info("Parsing [%d/%d]: %s", i, len(pdf_paths), filename)
            results.append(self.parse(path))
        return results
