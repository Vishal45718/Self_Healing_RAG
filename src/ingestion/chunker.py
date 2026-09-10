import hashlib
from typing import List
import logging

from src.schema import Document, Chunk

logger = logging.getLogger(__name__)

class RecursiveCharacterChunker:
    """Splits documents into smaller semantic chunks."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using the provided separators."""
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        separator = separators[0]
        for sep in separators:
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                break

        if separator == "":
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        splits = text.split(separator)
        good_splits = []
        for s in splits:
            if good_splits:
                good_splits[-1] += separator + s
            else:
                good_splits.append(s)

        final_chunks = []
        current_chunk = ""
        
        for split in splits:
            if not current_chunk:
                current_chunk = split
            elif len(current_chunk) + len(separator) + len(split) <= self.chunk_size:
                current_chunk += separator + split
            else:
                if len(current_chunk) > self.chunk_size:
                    # Further split current chunk if it exceeds the size
                    sub_chunks = self._split_text(current_chunk, separators[separators.index(separator) + 1:])
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(current_chunk)
                
                # Overlap logic
                # Go backwards through current_chunk to get the overlap
                overlap_text = current_chunk[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                
                # However, for true recursive splitting, typically we just append words.
                # A simple overlap strategy:
                current_chunk = overlap_text + separator + split if overlap_text else split
                
        if current_chunk:
            if len(current_chunk) > self.chunk_size:
                sub_chunks = self._split_text(current_chunk, separators[separators.index(separator) + 1:])
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(current_chunk)

        return final_chunks
        
    def _generate_chunk_id(self, document_id: str, chunk_index: int, content: str) -> str:
        """Generate a deterministic chunk ID."""
        hash_input = f"{document_id}_{chunk_index}_{content}".encode("utf-8")
        return hashlib.sha256(hash_input).hexdigest()

    def chunk_document(self, document: Document) -> List[Chunk]:
        """Split a single Document into Chunks."""
        if not document.content.strip():
            return []

        # Use a more standardized recursive text splitting approach.
        text_chunks = self._recursive_split(document.content, self.separators)
        
        chunks = []
        for i, text in enumerate(text_chunks):
            chunk_id = self._generate_chunk_id(document.id, i, text)
            chunks.append(Chunk(
                id=chunk_id,
                document_id=document.id,
                content=text,
                chunk_index=i,
                metadata=document.metadata.copy()
            ))
            
        return chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Standard recursive splitting approach handling overlap cleanly."""
        if len(text) <= self.chunk_size:
            return [text]

        separator = separators[-1]
        new_separators = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i+1:]
                break

        splits = text.split(separator) if separator else list(text)
        
        good_splits = []
        _separator = separator if separator else ""
        
        for s in splits:
            if len(s) > self.chunk_size:
                if good_splits:
                    good_splits.extend(self._recursive_split(s, new_separators))
                else:
                    good_splits.extend(self._recursive_split(s, new_separators))
            else:
                good_splits.append(s)

        # Merge splits with overlap
        merged = []
        current_chunk = []
        current_length = 0
        
        for s in good_splits:
            s_len = len(s) if not current_chunk else len(s) + len(_separator)
            
            if current_length + s_len > self.chunk_size and current_length > 0:
                merged.append(_separator.join(current_chunk))
                
                # Keep elements for overlap
                while current_length > self.chunk_overlap and len(current_chunk) > 1:
                    popped = current_chunk.pop(0)
                    current_length -= len(popped) + len(_separator)
                
                # If it's still too big, reset completely
                if current_length > self.chunk_overlap:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(s)
            current_length += s_len

        if current_chunk:
            merged.append(_separator.join(current_chunk))

        return merged

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Split a list of Documents into Chunks."""
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
