"""
PDF Scanner — V0: PDF Discovery Module

Recursively finds all PDF files within a given directory and reports
their filenames and full paths.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class PDFFile:
    """Represents a discovered PDF file."""
    filename: str
    full_path: str
    size_bytes: int


class PDFScanner:
    """Recursively scans a directory for PDF files."""

    PDF_EXTENSIONS = {".pdf"}

    def __init__(self, folder_path: str) -> None:
        """
        Initialize the scanner with a target folder path.

        Args:
            folder_path: Path to the directory to scan.

        Raises:
            FileNotFoundError: If the folder does not exist.
            NotADirectoryError: If the path exists but is not a directory.
        """
        self.folder_path = Path(folder_path).resolve()

        if not self.folder_path.exists():
            raise FileNotFoundError(
                f"The folder does not exist: {self.folder_path}"
            )
        if not self.folder_path.is_dir():
            raise NotADirectoryError(
                f"The path is not a directory: {self.folder_path}"
            )

    def scan(self) -> List[PDFFile]:
        """
        Recursively scan the folder for PDF files.

        Handles both lowercase (.pdf) and uppercase (.PDF) extensions,
        as well as mixed-case variants (.Pdf, .pDf, etc.).

        Returns:
            A sorted list of PDFFile objects found in the directory tree.
        """
        pdf_files: List[PDFFile] = []

        for root, _dirs, files in os.walk(self.folder_path):
            for filename in files:
                if Path(filename).suffix.lower() in self.PDF_EXTENSIONS:
                    full_path = os.path.join(root, filename)
                    try:
                        size = os.path.getsize(full_path)
                    except OSError:
                        size = 0
                    pdf_files.append(
                        PDFFile(
                            filename=filename,
                            full_path=str(Path(full_path).resolve()),
                            size_bytes=size,
                        )
                    )

        # Sort by filename for deterministic output
        pdf_files.sort(key=lambda f: f.filename.lower())
        return pdf_files


def format_size(size_bytes: int) -> str:
    """Format a file size in human-readable form."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def print_results(pdf_files: List[PDFFile]) -> None:
    """
    Pretty-print the list of discovered PDFs.

    Args:
        pdf_files: List of PDFFile objects to display.
    """
    if not pdf_files:
        print("\nNo PDF files found in the specified folder.")
        return

    count = len(pdf_files)
    label = "file" if count == 1 else "files"
    print(f"\nFound {count} PDF {label}.\n")

    for i, pdf in enumerate(pdf_files, start=1):
        print(f"{i}. {pdf.filename}")
        print(f"   {pdf.full_path}")
        print(f"   Size: {format_size(pdf.size_bytes)}")
        print()
