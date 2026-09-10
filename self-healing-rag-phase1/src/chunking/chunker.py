"""
Chunking.

Simple recursive character splitter: tries to break on paragraph, then
sentence, then hard character boundaries, in that order, respecting
chunk_size/overlap. This is intentionally the simplest strategy that
works well for general text — no need for anything fancier at this
project's scale (see DECISIONS.md).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import settings  # noqa: E402
from src.ingestion.loader import Document  # noqa: E402

_SPLIT_SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _split_text(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Recursively split text on the first separator that yields pieces
    small enough to fit chunk_size; falls back to hard char slicing."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    if not separators:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, rest_separators = separators[0], separators[1:]
    parts = [p for p in text.split(sep) if p.strip()]
    if len(parts) <= 1:
        return _split_text(text, chunk_size, rest_separators)

    chunks: list[str] = []
    buffer = ""
    for part in parts:
        candidate = f"{buffer}{sep}{part}" if buffer else part
        if len(candidate) <= chunk_size:
            buffer = candidate
        else:
            if buffer:
                chunks.append(buffer)
            if len(part) > chunk_size:
                chunks.extend(_split_text(part, chunk_size, rest_separators))
                buffer = ""
            else:
                buffer = part
    if buffer:
        chunks.append(buffer)
    return chunks


def _add_overlap(pieces: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or len(pieces) <= 1:
        return pieces
    overlapped = [pieces[0]]
    for prev, current in zip(pieces, pieces[1:]):
        tail = prev[-overlap:]
        overlapped.append(f"{tail}{current}")
    return overlapped


def chunk_document(
    document: Document,
    chunk_size: int = settings.CHUNK_SIZE,
    overlap: int = settings.CHUNK_OVERLAP,
) -> list[Chunk]:
    raw_pieces = _split_text(document.content, chunk_size, _SPLIT_SEPARATORS)
    pieces = _add_overlap(raw_pieces, overlap)

    chunks = []
    for idx, piece in enumerate(pieces):
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}_chunk{idx}",
                doc_id=document.doc_id,
                content=piece,
                chunk_index=idx,
                metadata={**document.metadata, "chunk_index": idx},
            )
        )
    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = settings.CHUNK_SIZE,
    overlap: int = settings.CHUNK_OVERLAP,
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, chunk_size=chunk_size, overlap=overlap))
    return all_chunks
