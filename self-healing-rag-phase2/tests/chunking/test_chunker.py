import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.chunking.chunker import chunk_document, chunk_documents  # noqa: E402
from src.ingestion.loader import Document  # noqa: E402


def make_doc(content: str, doc_id: str = "doc1") -> Document:
    return Document(doc_id=doc_id, content=content, metadata={"source": f"{doc_id}.txt"})


def test_document_smaller_than_chunk_size_produces_one_chunk():
    doc = make_doc("short text")
    chunks = chunk_document(doc, chunk_size=800, overlap=120)
    assert len(chunks) == 1
    assert chunks[0].content == "short text"
    assert chunks[0].chunk_id == "doc1_chunk0"


def test_empty_document_produces_no_chunks():
    doc = make_doc("")
    chunks = chunk_document(doc, chunk_size=800, overlap=120)
    assert chunks == []


def test_long_document_splits_into_multiple_chunks():
    paragraph = "This is a sentence about ants. " * 30  # ~960 chars
    doc = make_doc(paragraph)
    chunks = chunk_document(doc, chunk_size=200, overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        # overlap can push slightly over; allow small margin
        assert len(c.content) <= 200 + 20 + 5


def test_chunks_preserve_parent_metadata_and_add_chunk_index():
    doc = make_doc("word " * 200, doc_id="report")
    chunks = chunk_document(doc, chunk_size=100, overlap=10)
    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert c.doc_id == "report"
        assert c.metadata["source"] == "report.txt"
        assert c.metadata["chunk_index"] == i
        assert c.chunk_index == i


def test_chunk_documents_handles_multiple_docs():
    docs = [make_doc("word " * 200, doc_id="a"), make_doc("short", doc_id="b")]
    chunks = chunk_documents(docs, chunk_size=100, overlap=10)
    doc_ids = {c.doc_id for c in chunks}
    assert doc_ids == {"a", "b"}
