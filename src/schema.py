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

class RetrievalResult(BaseModel):
    """Represents a retrieved chunk with similarity score and distance."""
    id: str = Field(description="The unique identifier of the retrieved chunk.")
    content: str = Field(description="The text content of the retrieved chunk.")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata associated with the retrieved chunk."
    )
    distance: float = Field(
        description="Distance metric returned by vector store (e.g. cosine distance)."
    )
    similarity: float = Field(
        description="Calculated similarity score (1.0 - distance for cosine)."
    )

from typing import TypedDict, Optional, List
from src.critic.schema import CriticEvaluation

class GraphState(TypedDict):
    """Represents the state of the self-healing RAG workflow."""
    original_query: str
    current_query: str
    retrieved_chunks: List[RetrievalResult]
    generation: Optional[str]
    critic_evaluation: Optional[CriticEvaluation]
    iterations: int
