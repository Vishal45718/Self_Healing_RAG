import pytest
from pathlib import Path
from src.ingestion.loader import DocumentLoader

@pytest.fixture
def temp_files(tmp_path):
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Hello World", encoding="utf-8")
    
    md_file = tmp_path / "test.md"
    md_file.write_text("# Hello\nWorld", encoding="utf-8")
    
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")
    
    # We will skip PDF creation here and assume it's tested elsewhere or use a mock if needed, 
    # but the requirement states "verify the loader behavior with an actual PDF, not only mocks".
    # I'll create a minimal valid PDF byte string to write.
    pdf_file = tmp_path / "test.pdf"
    minimal_pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length 44 >>\nstream\n"
        b"BT\n/F1 24 Tf\n100 700 Td\n(Hello PDF) Tj\nET\n"
        b"endstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000243 00000 n \n"
        b"0000000338 00000 n \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\n"
        b"startxref\n426\n%%EOF\n"
    )
    pdf_file.write_bytes(minimal_pdf)
    
    corrupt_pdf = tmp_path / "corrupt.pdf"
    corrupt_pdf.write_bytes(b"not a pdf")
    
    unsupported = tmp_path / "image.png"
    unsupported.write_bytes(b"fake image data")

    return {
        "txt": txt_file,
        "md": md_file,
        "empty": empty_file,
        "pdf": pdf_file,
        "corrupt_pdf": corrupt_pdf,
        "unsupported": unsupported,
        "dir": tmp_path
    }

def test_load_txt(temp_files):
    doc = DocumentLoader.load_file(temp_files["txt"])
    assert doc is not None
    assert doc.id == "test.txt"
    assert doc.content == "Hello World"
    assert doc.metadata["file_type"] == "txt"
    assert "source" in doc.metadata

def test_load_md(temp_files):
    doc = DocumentLoader.load_file(temp_files["md"])
    assert doc is not None
    assert doc.id == "test.md"
    assert doc.content == "# Hello\nWorld"
    assert doc.metadata["file_type"] == "md"

def test_load_empty(temp_files):
    doc = DocumentLoader.load_file(temp_files["empty"])
    assert doc is not None
    assert doc.content == ""

def test_load_pdf(temp_files):
    doc = DocumentLoader.load_file(temp_files["pdf"])
    assert doc is not None
    assert doc.id == "test.pdf"
    assert "Hello PDF" in doc.content
    assert doc.metadata["file_type"] == "pdf"
    assert doc.metadata["total_pages"] == 1
    assert len(doc.metadata["page_info"]) == 1

def test_load_corrupt_pdf(temp_files):
    doc = DocumentLoader.load_file(temp_files["corrupt_pdf"])
    assert doc is None

def test_load_unsupported(temp_files):
    doc = DocumentLoader.load_file(temp_files["unsupported"])
    assert doc is None

def test_load_missing(temp_files):
    missing_file = temp_files["dir"] / "missing.txt"
    doc = DocumentLoader.load_file(missing_file)
    assert doc is None

def test_load_directory_mixed_batch(temp_files):
    docs = DocumentLoader.load_directory(temp_files["dir"])
    # Should load txt, md, empty, pdf. Should skip corrupt_pdf and unsupported.
    # Total = 4 valid documents.
    assert len(docs) == 4
    file_types = [doc.metadata["file_type"] for doc in docs]
    assert "txt" in file_types
    assert "md" in file_types
    assert "pdf" in file_types
