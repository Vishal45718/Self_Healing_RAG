import pytest
from src.schema import Document
from src.ingestion.chunker import RecursiveCharacterChunker

@pytest.fixture
def sample_document():
    return Document(
        id="doc_1",
        content="This is the first sentence.\n\nThis is the second sentence. It has more words.\n\nAnd the third sentence.",
        metadata={"source": "test.txt", "file_type": "txt"}
    )

def test_invalid_configuration():
    with pytest.raises(ValueError):
        RecursiveCharacterChunker(chunk_size=10, chunk_overlap=15)

def test_normal_document_chunking(sample_document):
    chunker = RecursiveCharacterChunker(chunk_size=40, chunk_overlap=10)
    chunks = chunker.chunk_document(sample_document)
    
    assert len(chunks) > 1
    assert chunks[0].document_id == "doc_1"
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert chunks[0].metadata["source"] == "test.txt"

def test_smaller_than_chunk_document(sample_document):
    chunker = RecursiveCharacterChunker(chunk_size=500, chunk_overlap=0)
    chunks = chunker.chunk_document(sample_document)
    
    assert len(chunks) == 1
    assert chunks[0].content == sample_document.content

def test_empty_document():
    doc = Document(id="empty", content="", metadata={})
    chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk_document(doc)
    
    assert len(chunks) == 0

def test_deterministic_ids(sample_document):
    chunker = RecursiveCharacterChunker(chunk_size=40, chunk_overlap=10)
    chunks1 = chunker.chunk_document(sample_document)
    chunks2 = chunker.chunk_document(sample_document)
    
    assert len(chunks1) == len(chunks2)
    for c1, c2 in zip(chunks1, chunks2):
        assert c1.id == c2.id

def test_overlap_logic():
    content = "A B C D E F G H I J K L M N O P"
    doc = Document(id="letters", content=content, metadata={})
    
    # Chunk size 10, overlap 4
    chunker = RecursiveCharacterChunker(chunk_size=10, chunk_overlap=4)
    chunks = chunker.chunk_document(doc)
    
    assert len(chunks) > 1
    # Check that there is an overlap of at least some characters (due to recursive split logic, it may not be exact)
    # But chunks[1] should start with some characters from the end of chunks[0]
    overlap_found = False
    for i in range(1, len(chunks)):
        words_prev = chunks[i-1].content.split()
        words_curr = chunks[i].content.split()
        if set(words_prev).intersection(set(words_curr)):
            overlap_found = True
            break
            
    assert overlap_found, "Expected some overlap between chunks"
