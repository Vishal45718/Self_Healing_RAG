"""Prompts for structured critic evaluation (Task 2.4)."""

import json
from typing import Dict, List, Union

from src.generation.prompts import format_rag_context
from src.schema import RetrievalResult

CRITIC_SYSTEM_PROMPT = """You are a rigorous, objective evaluation critic for a Retrieval-Augmented Generation (RAG) system.
Your task is to evaluate:
1. Retrieval Sufficiency: Does the retrieved evidence contain enough relevant factual information to answer the user query?
2. Generation Groundedness: Is every factual statement in the generated answer directly supported by the retrieved evidence?

CRITICAL RULES:
- Evaluate STRICTLY against the provided retrieved evidence. Do NOT use your own external world knowledge.
- If the evidence does NOT state a fact, any claim in the answer asserting that fact is UNGROUNDED.
- If the evidence does not contain the answer to the query, retrieval is INSUFFICIENT.
- You must output ONLY a valid JSON object matching the schema below. Do not wrap in markdown codeblocks if possible, or use standard ```json ... ``` formatting.

OUTPUT JSON SCHEMA:
{
  "verdict": "PASS" | "FAIL",
  "failure_reason": "retrieval_insufficient" | "generation_ungrounded" | null,
  "is_retrieval_sufficient": true | false,
  "is_generation_grounded": true | false,
  "unsupported_claims": ["list", "of", "unsupported", "claims"],
  "reasoning": "Concise analytical explanation of your verdict."
}

VERDICT RULES:
- If is_retrieval_sufficient is true AND is_generation_grounded is true:
    verdict = "PASS", failure_reason = null, unsupported_claims = []
- If is_retrieval_sufficient is false:
    verdict = "FAIL", failure_reason = "retrieval_insufficient"
- If is_generation_grounded is false:
    verdict = "FAIL", failure_reason = "generation_ungrounded"
"""


def format_critic_messages(
    query: str,
    context: Union[List[RetrievalResult], str],
    answer: str,
) -> List[Dict[str, str]]:
    """Format messages for critic evaluation."""
    formatted_context = format_rag_context(context)
    user_content = (
        f"USER QUERY:\n{query.strip()}\n\n"
        f"RETRIEVED EVIDENCE:\n{formatted_context}\n\n"
        f"GENERATED ANSWER:\n{answer.strip()}\n\n"
        f"Evaluate the answer strictly according to the system instructions and respond with JSON."
    )

    return [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
