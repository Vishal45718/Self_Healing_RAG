"""Generation package for Self-Healing RAG."""

from src.generation.generator import GenerationResult, Generator
from src.generation.llm_client import (
    BaseLLMClient,
    ChatChoice,
    ChatCompletionResponse,
    ChatMessage,
    GeminiLLMClient,
    HFInferenceAdapter,
    get_llm_client,
)
from src.generation.prompts import RAG_SYSTEM_PROMPT, format_rag_messages

__all__ = [
    "Generator",
    "GenerationResult",
    "RAG_SYSTEM_PROMPT",
    "format_rag_messages",
    "BaseLLMClient",
    "GeminiLLMClient",
    "HFInferenceAdapter",
    "get_llm_client",
    "ChatMessage",
    "ChatChoice",
    "ChatCompletionResponse",
]
