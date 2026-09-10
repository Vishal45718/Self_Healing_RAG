"""
Single source of truth for configuration.

Deliberately flat: no provider registry, no factory pattern.
To change a model/provider, edit the .env values below — not code.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "data" / "chroma_store"
LOG_DIR = PROJECT_ROOT / "logs"

# --- Ingestion ---
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))          # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))    # characters

# --- Embeddings ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- LLM (Hugging Face Inference Providers) ---
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_PROVIDER = os.getenv("HF_PROVIDER", "auto")             # e.g. "auto", "together", "fireworks-ai"
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

# --- Self-healing graph ---
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))
