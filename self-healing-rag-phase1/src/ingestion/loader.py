"""
Document ingestion.

Loads .txt, .md, and .pdf files into a uniform Document representation.
Invalid or unsupported files are skipped and reported, never allowed to
crash the whole batch load.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pypdf import PdfReader

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import settings  # noqa: E402


@dataclass
class Document:
    doc_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadError:
    path: str
    reason: str


@dataclass
class LoadResult:
    documents: list[Document]
    errors: list[LoadError]


def _load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_pdf_file(path: Path) -> tuple[str, int]:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages), len(reader.pages)


def load_document(path: Path) -> Document:
    """
    Load a single file. Raises ValueError on unsupported extension,
    FileNotFoundError if missing, or the underlying read/parse error
    for corrupt files. Callers doing batch loads should catch these
    (see load_documents) rather than let one bad file abort the run.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext not in settings.SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    metadata: dict[str, Any] = {
        "source": str(path),
        "file_type": ext.lstrip("."),
    }

    if ext in (".txt", ".md"):
        content = _load_text_file(path)
    elif ext == ".pdf":
        content, page_count = _load_pdf_file(path)
        metadata["page_count"] = page_count
    else:  # pragma: no cover - guarded above
        raise ValueError(f"Unsupported file type: {ext}")

    if not content.strip():
        raise ValueError(f"File has no extractable text content: {path}")

    return Document(doc_id=path.stem, content=content, metadata=metadata)


def load_documents(paths: list[Path]) -> LoadResult:
    """
    Batch-load documents. Never raises on a per-file failure — instead
    collects failures in LoadResult.errors so one bad file doesn't
    abort ingestion of the rest.
    """
    documents: list[Document] = []
    errors: list[LoadError] = []

    for path in paths:
        try:
            documents.append(load_document(path))
        except Exception as exc:  # noqa: BLE001 - intentionally broad, logged not swallowed
            errors.append(LoadError(path=str(path), reason=str(exc)))

    return LoadResult(documents=documents, errors=errors)


def load_directory(directory: Path) -> LoadResult:
    """Load every file in a directory (non-recursive)."""
    paths = sorted(p for p in directory.iterdir() if p.is_file())
    return load_documents(paths)
