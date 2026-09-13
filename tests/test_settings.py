"""Unit tests for configuration and settings."""

from config.settings import Settings


def test_default_settings():
    """Verify default settings values."""
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "gemini"
    assert settings.gemini_model_id == "gemini-3.6-flash"
    assert settings.gemini_api_key is None
    assert settings.active_model_id == "gemini-3.6-flash"
    assert settings.hf_provider == "together"
    assert settings.llm_model_id == "meta-llama/Llama-3.3-70B-Instruct"
    assert settings.embedding_model_id == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.chroma_path == "./data/chroma"
    assert settings.max_retries == 3
    assert settings.hf_token is None


def test_settings_env_override(monkeypatch):
    """Verify settings can be overridden via environment variables."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_MODEL_ID", "gemini-3.6-flash")
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.setenv("HF_PROVIDER", "featherless")
    monkeypatch.setenv("LLM_MODEL_ID", "Qwen/Qwen2.5-72B-Instruct")
    monkeypatch.setenv("MAX_RETRIES", "5")
    monkeypatch.setenv("CHROMA_PATH", "./custom/chroma")

    settings = Settings()
    assert settings.llm_provider == "gemini"
    assert settings.gemini_model_id == "gemini-3.6-flash"
    assert settings.gemini_api_key == "mock-gemini-key"
    assert settings.active_model_id == "gemini-3.6-flash"
    assert settings.hf_provider == "featherless"
    assert settings.llm_model_id == "Qwen/Qwen2.5-72B-Instruct"
    assert settings.max_retries == 5
    assert settings.chroma_path == "./custom/chroma"

    # Test active_model_id when LLM_PROVIDER is huggingface
    monkeypatch.setenv("LLM_PROVIDER", "huggingface")
    settings_hf = Settings()
    assert settings_hf.active_model_id == "Qwen/Qwen2.5-72B-Instruct"
