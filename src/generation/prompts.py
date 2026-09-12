"""Prompt templates and formatting for baseline RAG generation (Task 2.3)."""

from typing import Dict, List, Optional, Union

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
The user's original query did not retrieve sufficient or relevant context.
Your task is to rewrite the original query to improve retrieval.
You may use the critic's reasoning to guide your reformulation.

CRITICAL RULES:
- Output ONLY the new query text — no quotes, preambles, or conversational text.
- Your reformulated query MUST be meaningfully different from every query listed in the
  'Previously Tried Queries' section below.  Repeating a previous query verbatim is forbidden.
- Preserve the original intent of the question while broadening or sharpening the vocabulary.
"""


def format_reformulate_messages(
    original_query: str,
    critic_reasoning: str,
    query_history: Optional[List[str]] = None,
    system_prompt: str = REFORMULATE_SYSTEM_PROMPT,
) -> List[Dict[str, str]]:
    """Format original query and critic reasoning into chat completion messages for query reformulation.

    Args:
        original_query: The user's original question.
        critic_reasoning: The feedback from the critic explaining why the previous attempt failed.
        query_history: All queries already tried (including the original). Used to prevent
            repetition of previous queries.
        system_prompt: The system instruction prompt.

    Returns:
        A list of role/content dictionaries for the chat completion API.
    """
    history_section = ""
    if query_history:
        history_lines = "\n".join(f"  - {q}" for q in query_history)
        history_section = f"Previously Tried Queries (DO NOT repeat any of these):\n{history_lines}\n\n"

    user_content = (
        f"Original Query: {original_query.strip()}\n\n"
        f"{history_section}"
        f"Critic Feedback (Why it failed): {critic_reasoning.strip()}\n\n"
        f"Please reformulate the query to improve retrieval."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


REGENERATE_SYSTEM_PROMPT = """You are a strict, factual question-answering assistant.
A previous answer you generated was evaluated and found to be UNGROUNDED — it contained
claims not supported by the provided evidence.

Your task is to generate a corrected, fully-grounded answer using ONLY the retrieved context.

CRITICAL RULES:
1. Every factual claim you make MUST be directly and explicitly supported by the retrieved context.
2. Do NOT include any of the previously identified unsupported claims listed below.
3. If the context genuinely does not support a complete answer, state clearly:
   "The provided context is insufficient to answer this question."
4. Do NOT invent, extrapolate, or assume any facts.
"""


def format_regenerate_messages(
    query: str,
    context: Union[List[RetrievalResult], str],
    critic_reasoning: str,
    unsupported_claims: Optional[List[str]] = None,
    system_prompt: str = REGENERATE_SYSTEM_PROMPT,
) -> List[Dict[str, str]]:
    """Format messages for a strict re-generation when the previous answer was ungrounded.

    This reuses the same retrieved context chunks — NO new retrieval is performed.
    The prompt embeds the critic's reasoning and the specific unsupported claims so
    the model knows exactly what to avoid.

    Args:
        query: The original user query.
        context: The same retrieved context chunks used in the failed generation.
        critic_reasoning: The critic's explanation of why the previous answer failed.
        unsupported_claims: Specific claims from the previous answer that lacked grounding.
        system_prompt: The system instruction prompt.

    Returns:
        A list of role/content dictionaries for the chat completion API.
    """
    formatted_context = format_rag_context(context)

    claims_section = ""
    if unsupported_claims:
        claims_lines = "\n".join(f"  - {c}" for c in unsupported_claims)
        claims_section = f"\nSpecific Unsupported Claims to Avoid:\n{claims_lines}\n"

    user_content = (
        f"Retrieved Context:\n"
        f"-----------------\n"
        f"{formatted_context}\n"
        f"-----------------\n\n"
        f"Question: {query.strip()}\n\n"
        f"Critic Feedback (Why the previous answer failed):\n{critic_reasoning.strip()}"
        f"{claims_section}\n"
        f"Please generate a corrected, fully-grounded answer."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
