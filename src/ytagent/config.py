"""Model/provider resolution. This module is framework-light on purpose:
everything provider-specific is a `provider:model` string handed to
LangChain's init_chat_model, so swapping vendors never touches agent code.

Resolution order for the model string:
    1. --model flag on the CLI
    2. YTAGENT_MODEL environment variable
API keys come from each provider's standard env var (OPENAI_API_KEY,
ANTHROPIC_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY; Ollama needs none).
"""

from __future__ import annotations

import importlib.util
import os

ENV_MODEL = "YTAGENT_MODEL"

# provider prefix -> (pip package, api-key env var or None)
KNOWN_PROVIDERS: dict[str, tuple[str, str | None]] = {
    "openai": ("langchain-openai", "OPENAI_API_KEY"),
    "anthropic": ("langchain-anthropic", "ANTHROPIC_API_KEY"),
    "google_genai": ("langchain-google-genai", "GOOGLE_API_KEY"),
    "groq": ("langchain-groq", "GROQ_API_KEY"),
    "ollama": ("langchain-ollama", None),
}

# provider prefix -> extra name in pyproject.toml
PROVIDER_EXTRAS = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google_genai": "google",
    "groq": "groq",
    "ollama": "ollama",
}


def install_hint(provider: str) -> str:
    extra = PROVIDER_EXTRAS.get(provider)
    package = KNOWN_PROVIDERS[provider][0]
    return f"uv sync --extra {extra}" if extra else f"uv add {package}"

_IMPORT_NAMES = {
    "langchain-openai": "langchain_openai",
    "langchain-anthropic": "langchain_anthropic",
    "langchain-google-genai": "langchain_google_genai",
    "langchain-groq": "langchain_groq",
    "langchain-ollama": "langchain_ollama",
}


class ConfigError(Exception):
    pass


def resolve_model_string(flag_value: str | None) -> str:
    model = flag_value or os.environ.get(ENV_MODEL)
    if not model:
        raise ConfigError(
            "no model configured. Pass --model provider:model "
            f"(e.g. --model ollama:llama3.1, --model groq:llama-3.3-70b-versatile) "
            f"or set the {ENV_MODEL} environment variable."
        )
    return model


def preflight(model: str) -> None:
    """Fail fast with an actionable message instead of a deep import error."""
    if ":" not in model:
        raise ConfigError(
            f"model '{model}' has no provider prefix; use provider:model, "
            f"e.g. one of: {', '.join(sorted(KNOWN_PROVIDERS))}"
        )
    provider = model.split(":", 1)[0]
    if provider not in KNOWN_PROVIDERS:
        # Unknown to us but maybe known to init_chat_model — let it try.
        return
    package, key_env = KNOWN_PROVIDERS[provider]
    import_name = _IMPORT_NAMES[package]
    if importlib.util.find_spec(import_name) is None:
        raise ConfigError(
            f"provider '{provider}' needs the {package} package: run `{install_hint(provider)}`"
        )
    if key_env and not os.environ.get(key_env):
        raise ConfigError(f"provider '{provider}' requires the {key_env} environment variable")


def make_model(model_string: str, temperature: float = 0.0):
    """Instantiate a chat model from a provider:model string."""
    preflight(model_string)
    from langchain.chat_models import init_chat_model

    return init_chat_model(model_string, temperature=temperature)


def provider_status() -> list[tuple[str, str, bool, bool]]:
    """(provider, install hint, installed?, key present or not needed?) for `ytagent models`."""
    rows = []
    for provider, (package, key_env) in sorted(KNOWN_PROVIDERS.items()):
        installed = importlib.util.find_spec(_IMPORT_NAMES[package]) is not None
        key_ok = key_env is None or bool(os.environ.get(key_env))
        rows.append((provider, install_hint(provider), installed, key_ok))
    return rows
