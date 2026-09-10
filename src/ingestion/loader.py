from pathlib import Path
from typing import List, Optional
import pypdf
import logging

from src.schema import Document

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Loads documents from various file formats into Document models."""

    @staticmethod
    def load_txt(file_path: Path) -> Optional[Document]:
        try:
            content = file_path.read_text(encoding="utf-8")
            return Document(
                id=file_path.name,
                content=content,
                metadata={"source": str(file_path), "file_type": "txt"}
            )
        except Exception as e:
            logger.error(f"Failed to load TXT {file_path}: {e}")
            return None

    @staticmethod
    def load_md(file_path: Path) -> Optional[Document]:
        try:
            content = file_path.read_text(encoding="utf-8")
            return Document(
                id=file_path.name,
                content=content,
                metadata={"source": str(file_path), "file_type": "md"}
            )
        except Exception as e:
            logger.error(f"Failed to load MD {file_path}: {e}")
            return None

    @staticmethod
    def load_pdf(file_path: Path) -> Optional[Document]:
        try:
            content_parts = []
            page_info = []
            
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        content_parts.append(text)
                        page_info.append({"page": i + 1, "length": len(text)})
            
            return Document(
                id=file_path.name,
                content="\n".join(content_parts),
                metadata={
                    "source": str(file_path),
                    "file_type": "pdf",
                    "page_info": page_info,
                    "total_pages": len(reader.pages)
                }
            )
        except Exception as e:
            logger.error(f"Failed to load PDF {file_path}: {e}")
            return None

    @classmethod
    def load_file(cls, file_path: Path) -> Optional[Document]:
        """Load a single file based on its extension."""
        if not file_path.exists() or not file_path.is_file():
            logger.warning(f"File not found or invalid: {file_path}")
            return None

        ext = file_path.suffix.lower()
        if ext == ".txt":
            return cls.load_txt(file_path)
        elif ext == ".md":
            return cls.load_md(file_path)
        elif ext == ".pdf":
            return cls.load_pdf(file_path)
        else:
            logger.warning(f"Unsupported file type for {file_path}")
            return None

    @classmethod
    def load_directory(cls, dir_path: Path) -> List[Document]:
        """Load all supported documents from a directory."""
        documents = []
        if not dir_path.exists() or not dir_path.is_dir():
            logger.error(f"Directory not found: {dir_path}")
            return documents
        
        for file_path in dir_path.iterdir():
            if file_path.is_file():
                doc = cls.load_file(file_path)
                if doc:
                    documents.append(doc)
        
        return documents
