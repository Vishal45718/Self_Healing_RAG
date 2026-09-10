# Self-Healing RAG

A robust, self-correcting Retrieval-Augmented Generation (RAG) system built with LangGraph, ChromaDB, Sentence-Transformers, and Hugging Face Inference Providers.

## Overview

Self-Healing RAG overcomes common failure modes of traditional RAG pipelines (such as retrieval of irrelevant context, hallucinated answers, or missing information) by introducing cyclical evaluation and correction loops:
1. **Retrieve**: Retrieve candidate document chunks from Chroma.
2. **Critique**: Assess document relevance and answer faithfulness.
3. **Reformulate**: Intelligently rewrite search queries if retrieval fails or is insufficient.
4. **Generate**: Produce grounded, verifiable responses.

## Project Structure

```
├── config/             # Centralized configuration (settings.py)
├── data/               # Local data storage and Chroma persistence
├── logs/               # Application run logs
├── src/                # Core implementation packages
├── tests/              # Automated test suite
├── .env.example        # Environment variable template
├── AGENTS.md           # Agent protocols and conventions
├── BUG_LOG.md          # Bug tracking and resolution log
├── DECISIONS.md        # Architectural decision records
├── PROJECT_PLAN.md     # Phase-by-phase project roadmap
└── pyproject.toml      # Build configuration and dependencies
```

## Quick Start

### 1. Environment Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configuration

Copy the example environment file and add your credentials:

```bash
cp .env.example .env
```

Edit `.env` to configure:
- `HF_TOKEN`: Your Hugging Face user access token.
- `HF_PROVIDER`: Selected provider (e.g. `together`, `featherless`).
- `LLM_MODEL_ID`: Hugging Face model repository ID.
- `EMBEDDING_MODEL_ID`: Embedding model name or HF repo ID.
- `CHROMA_PATH`: Path for local Chroma storage.
- `MAX_RETRIES`: Maximum self-healing correction attempts.
