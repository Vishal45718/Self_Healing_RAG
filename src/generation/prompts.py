"""Prompt templates and formatting for baseline RAG generation (Task 2.3)."""

from typing import Dict, List, Union

from src.schema import RetrievalResult

RAG_SYSTEM_PROMPT = """You are a factual, concise question-answering assistant.
Answer the user's question using ONLY the provided retrieved context.

Rules:
1. Base your answer strictly and entirely on the provided retrieved context.
2. Do NOT invent, extrapolate, assume, or incorporate any facts not directly stated in the context.
3. If the provided context is empty, irrelevant, or insufficient to answer the question, state clearly: "The provided context is insufficient to answer this question."
"""


def format_rag_context(context: Union[List[RetrievalResult], str]) -> str:
    """Format retrieval results or raw text into a clean context block."""
    if isinstance(context, str):
        cleaned = context.strip()
        return cleaned if cleaned else "None provided."

    if not context:
        return "None provided."

    chunks_text = []
    for idx, item in enumerate(context, 1):
        chunks_text.append(f"[{idx}] (ID: {item.id})\n{item.content.strip()}")
    return "\n\n".join(chunks_text)


def format_rag_messages(
    query: str,
    context: Union[List[RetrievalResult], str],
    system_prompt: str = RAG_SYSTEM_PROMPT,
) -> List[Dict[str, str]]:
    """Format user query and retrieved context into chat completion messages.

    Args:
        query: The user's question.
        context: List of RetrievalResult objects or a raw context string.
        system_prompt: The system instruction prompt.

    Returns:
        A list of role/content dictionaries for the chat completion API.
    """
    formatted_context = format_rag_context(context)
    user_content = (
        f"Retrieved Context:\n"
        f"-----------------\n"
        f"{formatted_context}\n"
        f"-----------------\n\n"
        f"Question: {query.strip()}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


REFORMULATE_SYSTEM_PROMPT = """You are an expert query reformulator for a semantic search engine.
The user's original query did not retrieve sufficient or relevant context, or the generated answer was ungrounded.
Your task is to rewrite the original query to improve retrieval. 
You may use the critic's reasoning to guide your reformulation.
Output ONLY the new query text, without any quotes, preambles, or conversational text.
"""


def format_reformulate_messages(
    original_query: str,
    critic_reasoning: str,
    system_prompt: str = REFORMULATE_SYSTEM_PROMPT,
) -> List[Dict[str, str]]:
    """Format original query and critic reasoning into chat completion messages for query reformulation.

    Args:
        original_query: The user's original question.
        critic_reasoning: The feedback from the critic explaining why the previous attempt failed.
        system_prompt: The system instruction prompt.

    Returns:
        A list of role/content dictionaries for the chat completion API.
    """
    user_content = (
        f"Original Query: {original_query.strip()}\n\n"
        f"Critic Feedback (Why it failed): {critic_reasoning.strip()}\n\n"
        f"Please reformulate the query to improve retrieval."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
