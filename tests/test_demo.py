"""Unit and regression tests for the demo runner script."""

import pytest
from demo import run_demo, OfflineDeterministicClient


def test_offline_deterministic_client():
    """Verify that the mock offline client behaves deterministically across prompt types."""
    client = OfflineDeterministicClient()

    # 1. Standard Generation
    messages = [
        {"role": "system", "content": "You are a factual, concise question-answering assistant."},
        {"role": "user", "content": "Question: What is ChromaDB?"},
    ]
    res = client.chat_completion(messages)
    assert "ChromaDB is an open-source" in res.choices[0].message.content

    # 2. Hallucination on attempt 1
    messages_recovery = [
        {"role": "system", "content": "You are a factual, concise question-answering assistant."},
        {"role": "user", "content": "Question: How does the recovery workflow operate?"},
    ]
    res_recovery = client.chat_completion(messages_recovery)
    assert "quantum annealing" in res_recovery.choices[0].message.content

    # 3. Critic rejects hallucination
    messages_critic = [
        {"role": "system", "content": "You are a rigorous, objective evaluation critic for a RAG system.\nOUTPUT JSON SCHEMA:\n{}"},
        {"role": "user", "content": "The answer mentions quantum annealing hardware and assembly."},
    ]
    res_critic = client.chat_completion(messages_critic)
    assert '"verdict": "FAIL"' in res_critic.choices[0].message.content
    assert "generation_ungrounded" in res_critic.choices[0].message.content

    # 4. Regeneration after ungrounded feedback
    messages_regen = [
        {"role": "system", "content": "A previous answer you generated was evaluated and found to be ungrounded."},
        {"role": "user", "content": "Question: How does the recovery workflow operate?"},
    ]
    res_regen = client.chat_completion(messages_regen)
    assert "Self-Healing RAG coordinates retrieval" in res_regen.choices[0].message.content

    # 5. Abstention on unanswerable query
    messages_abstain = [
        {"role": "system", "content": "You are a factual, concise question-answering assistant."},
        {"role": "user", "content": "Question: What is the secret recipe for dark matter propulsion?"},
    ]
    res_abstain = client.chat_completion(messages_abstain)
    assert "does not contain sufficient information" in res_abstain.choices[0].message.content


def test_run_demo_offline_end_to_end():
    """Verify that the full demo runs end-to-end without raising exceptions."""
    run_demo(is_live=False)
