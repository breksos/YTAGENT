import pytest

from ytagent.config import ConfigError, ENV_MODEL, preflight, resolve_model_string


def test_flag_beats_env(monkeypatch):
    monkeypatch.setenv(ENV_MODEL, "ollama:from-env")
    assert resolve_model_string("groq:from-flag") == "groq:from-flag"


def test_env_fallback(monkeypatch):
    monkeypatch.setenv(ENV_MODEL, "ollama:from-env")
    assert resolve_model_string(None) == "ollama:from-env"


def test_no_model_is_actionable_error(monkeypatch):
    monkeypatch.delenv(ENV_MODEL, raising=False)
    with pytest.raises(ConfigError, match="--model"):
        resolve_model_string(None)


def test_preflight_requires_provider_prefix():
    with pytest.raises(ConfigError, match="provider:model"):
        preflight("llama3.1")


def test_preflight_lets_unknown_providers_through():
    preflight("someday_new_provider:model-x")  # init_chat_model gets to decide


def test_preflight_missing_package_message(monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ConfigError, match="uv sync --extra groq"):
        preflight("groq:llama-3.3-70b-versatile")


def test_preflight_missing_api_key(monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        preflight("groq:llama-3.3-70b-versatile")
