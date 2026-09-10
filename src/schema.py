from typing import Any, Dict
from pydantic import BaseModel, Field

class Document(BaseModel):
    """Represents a loaded document before chunking."""
    id: str = Field(description="A unique identifier for the document.")
    content: str = Field(description="The full text content of the document.")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata associated with the document (e.g., source, file_type)."
    )

class Chunk(BaseModel):
    """Represents a text chunk split from a Document."""
    id: str = Field(description="A deterministic identifier for the chunk.")
    document_id: str = Field(description="The ID of the parent document.")
    content: str = Field(description="The text content of the chunk.")
    chunk_index: int = Field(description="The sequential index of this chunk within the document.")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata inherited from the document and specific to this chunk."
    )
