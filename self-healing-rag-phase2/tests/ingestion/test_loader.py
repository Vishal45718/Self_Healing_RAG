import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingestion.loader import load_directory, load_document, load_documents  # noqa: E402

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "data" / "sample_docs"


def test_load_valid_txt_file():
    doc = load_document(SAMPLE_DIR / "note.txt")
    assert "Eiffel Tower" in doc.content
    assert doc.metadata["file_type"] == "txt"
    assert doc.metadata["source"].endswith("note.txt")


def test_load_valid_markdown_file():
    doc = load_document(SAMPLE_DIR / "readme.md")
    assert "markdown sample" in doc.content
    assert doc.metadata["file_type"] == "md"


def test_load_unsupported_extension_raises():
    try:
        load_document(SAMPLE_DIR / "broken.xyz")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Unsupported file type" in str(exc)


def test_load_empty_file_raises():
    try:
        load_document(SAMPLE_DIR / "empty.txt")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "no extractable text" in str(exc)


def test_load_missing_file_raises():
    try:
        load_document(SAMPLE_DIR / "does_not_exist.txt")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_batch_load_mixed_valid_and_invalid_does_not_crash():
    paths = [
        SAMPLE_DIR / "note.txt",
        SAMPLE_DIR / "readme.md",
        SAMPLE_DIR / "broken.xyz",
        SAMPLE_DIR / "empty.txt",
    ]
    result = load_documents(paths)
    assert len(result.documents) == 2
    assert len(result.errors) == 2
    loaded_ids = {d.doc_id for d in result.documents}
    assert loaded_ids == {"note", "readme"}


def test_load_directory_loads_supported_and_reports_rest():
    result = load_directory(SAMPLE_DIR)
    assert len(result.documents) == 2
    assert len(result.errors) == 2
