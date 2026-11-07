"""LLM factory: the only place that knows which provider is in use.

The rest of the project calls get_llm("main") or get_llm("fast") and never
imports a provider class directly, so switching Groq <-> Gemini is a
one-line change in .env.
"""
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from hr_assistant.config import Settings, get_settings

Role = Literal["main", "fast"]


class MissingAPIKeyError(RuntimeError):
    pass


def _resolve(settings: Settings, role: Role) -> tuple[str, str, str]:
    """Return (model_name, provider, api_key) for the configured provider."""
    if settings.llm_provider == "groq":
        key = settings.groq_api_key
        model = settings.groq_main_model if role == "main" else settings.groq_fast_model
        key_name = "GROQ_API_KEY"
    else:
        key = settings.google_api_key
        model = settings.gemini_main_model if role == "main" else settings.gemini_fast_model
        key_name = "GOOGLE_API_KEY"

    if not key:
        raise MissingAPIKeyError(
            f"{key_name} is not set. Add it to .env (see .env.example)."
        )
    return model, settings.llm_provider, key


def get_llm(role: Role = "main", settings: Settings | None = None) -> BaseChatModel:
    settings = settings or get_settings()
    model, provider, key = _resolve(settings, role)
    extra = {}
    if provider == "groq" and settings.groq_reasoning_effort:
        extra["reasoning_effort"] = settings.groq_reasoning_effort
    return init_chat_model(
        model,
        model_provider=provider,
        api_key=key,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
        **extra,
    )


def describe(settings: Settings | None = None) -> dict:
    """Human-readable summary of the active configuration (no secrets)."""
    s = settings or get_settings()
    if s.llm_provider == "groq":
        return {"provider": "groq", "main": s.groq_main_model, "fast": s.groq_fast_model}
    return {"provider": "google_genai", "main": s.gemini_main_model, "fast": s.gemini_fast_model}
