"""
Tests for the PDF Scanner module (V0 — PDF Discovery).
"""

import os
import tempfile
from pathlib import Path

import pytest

from src.pdf_scanner import PDFFile, PDFScanner, format_size, print_results


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir_with_pdfs(tmp_path: Path):
    """Create a temp directory tree with some .pdf files."""
    # root-level PDFs
    (tmp_path / "paper1.pdf").write_bytes(b"%PDF-1.4 fake content one")
    (tmp_path / "paper2.PDF").write_bytes(b"%PDF-1.4 fake content two")  # uppercase ext

    # nested sub-folder
    sub = tmp_path / "subfolder"
    sub.mkdir()
    (sub / "nested.pdf").write_bytes(b"%PDF-1.4 nested content")

    # non-PDF file (should be ignored)
    (tmp_path / "readme.txt").write_text("not a pdf")
    (tmp_path / "image.png").write_bytes(b"\x89PNG fake")

    return tmp_path


@pytest.fixture
def tmp_dir_empty(tmp_path: Path):
    """Create a temp directory with no PDFs."""
    (tmp_path / "notes.txt").write_text("no pdfs here")
    return tmp_path


@pytest.fixture
def tmp_dir_only_pdfs(tmp_path: Path):
    """Create a temp directory with exactly one PDF."""
    (tmp_path / "single.pdf").write_bytes(b"%PDF-1.4 solo")
    return tmp_path


# ---------------------------------------------------------------------------
# Tests: PDFScanner.__init__
# ---------------------------------------------------------------------------

class TestScannerInit:

    def test_valid_directory(self, tmp_dir_with_pdfs):
        scanner = PDFScanner(str(tmp_dir_with_pdfs))
        assert scanner.folder_path == tmp_dir_with_pdfs.resolve()

    def test_nonexistent_directory(self):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            PDFScanner("/absolutely/nonexistent/path/xyz123")

    def test_path_is_file_not_directory(self, tmp_path):
        file_path = tmp_path / "somefile.txt"
        file_path.write_text("hello")
        with pytest.raises(NotADirectoryError, match="not a directory"):
            PDFScanner(str(file_path))


# ---------------------------------------------------------------------------
# Tests: PDFScanner.scan
# ---------------------------------------------------------------------------

class TestScannerScan:

    def test_finds_all_pdfs_recursively(self, tmp_dir_with_pdfs):
        scanner = PDFScanner(str(tmp_dir_with_pdfs))
        results = scanner.scan()

        filenames = [r.filename for r in results]
        assert len(results) == 3
        assert "paper1.pdf" in filenames
        assert "paper2.PDF" in filenames
        assert "nested.pdf" in filenames

    def test_handles_uppercase_extensions(self, tmp_dir_with_pdfs):
        scanner = PDFScanner(str(tmp_dir_with_pdfs))
        results = scanner.scan()
        uppercase_files = [r for r in results if r.filename == "paper2.PDF"]
        assert len(uppercase_files) == 1

    def test_returns_empty_list_for_no_pdfs(self, tmp_dir_empty):
        scanner = PDFScanner(str(tmp_dir_empty))
        results = scanner.scan()
        assert results == []

    def test_single_pdf(self, tmp_dir_only_pdfs):
        scanner = PDFScanner(str(tmp_dir_only_pdfs))
        results = scanner.scan()
        assert len(results) == 1
        assert results[0].filename == "single.pdf"

    def test_full_path_is_absolute(self, tmp_dir_with_pdfs):
        scanner = PDFScanner(str(tmp_dir_with_pdfs))
        results = scanner.scan()
        for r in results:
            assert os.path.isabs(r.full_path)

    def test_size_is_positive(self, tmp_dir_with_pdfs):
        scanner = PDFScanner(str(tmp_dir_with_pdfs))
        results = scanner.scan()
        for r in results:
            assert r.size_bytes > 0

    def test_results_are_sorted_by_filename(self, tmp_dir_with_pdfs):
        scanner = PDFScanner(str(tmp_dir_with_pdfs))
        results = scanner.scan()
        filenames = [r.filename.lower() for r in results]
        assert filenames == sorted(filenames)

    def test_ignores_non_pdf_files(self, tmp_dir_with_pdfs):
        scanner = PDFScanner(str(tmp_dir_with_pdfs))
        results = scanner.scan()
        for r in results:
            assert r.filename.lower().endswith(".pdf")

    def test_empty_directory(self, tmp_path):
        scanner = PDFScanner(str(tmp_path))
        results = scanner.scan()
        assert results == []


# ---------------------------------------------------------------------------
# Tests: PDFFile dataclass
# ---------------------------------------------------------------------------

class TestPDFFile:

    def test_attributes(self):
        pdf = PDFFile(filename="test.pdf", full_path="/a/b/test.pdf", size_bytes=1024)
        assert pdf.filename == "test.pdf"
        assert pdf.full_path == "/a/b/test.pdf"
        assert pdf.size_bytes == 1024


# ---------------------------------------------------------------------------
# Tests: format_size
# ---------------------------------------------------------------------------

class TestFormatSize:

    def test_bytes(self):
        assert format_size(500) == "500 B"

    def test_kilobytes(self):
        assert format_size(2048) == "2.0 KB"

    def test_megabytes(self):
        assert format_size(5 * 1024 * 1024) == "5.0 MB"

    def test_zero(self):
        assert format_size(0) == "0 B"


# ---------------------------------------------------------------------------
# Tests: print_results
# ---------------------------------------------------------------------------

class TestPrintResults:

    def test_prints_no_pdfs_message(self, capsys):
        print_results([])
        captured = capsys.readouterr()
        assert "No PDF files found" in captured.out

    def test_prints_count_and_files(self, capsys):
        files = [
            PDFFile("a.pdf", "/x/a.pdf", 100),
            PDFFile("b.pdf", "/x/b.pdf", 200),
        ]
        print_results(files)
        captured = capsys.readouterr()
        assert "Found 2 PDF files" in captured.out
        assert "a.pdf" in captured.out
        assert "b.pdf" in captured.out

    def test_singular_label(self, capsys):
        files = [PDFFile("only.pdf", "/x/only.pdf", 50)]
        print_results(files)
        captured = capsys.readouterr()
        assert "Found 1 PDF file." in captured.out


# ---------------------------------------------------------------------------
# Tests: main() integration
# ---------------------------------------------------------------------------

class TestMainIntegration:

    def test_main_with_valid_folder(self, tmp_dir_with_pdfs):
        from src.main import main
        exit_code = main(["--folder", str(tmp_dir_with_pdfs)])
        assert exit_code == 0

    def test_main_with_invalid_folder(self):
        from src.main import main
        exit_code = main(["--folder", "/nonexistent/folder/xyz"])
        assert exit_code == 1

    def test_main_with_empty_folder(self, tmp_dir_empty):
        from src.main import main
        exit_code = main(["--folder", str(tmp_dir_empty)])
        assert exit_code == 0  # Not an error — just no PDFs found
