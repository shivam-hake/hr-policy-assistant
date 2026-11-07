"""Unit tests for the LLM factory. No network calls: we only check that the
right provider and model get selected. These run in CI for free."""
import pytest

from hr_assistant.config import Settings
from hr_assistant.llm import MissingAPIKeyError, get_llm


def test_groq_main_model_selected():
    s = Settings(_env_file=None, llm_provider="groq", groq_api_key="test-key")
    llm = get_llm("main", s)
    assert type(llm).__name__ == "ChatGroq"
    assert llm.model_name == s.groq_main_model


def test_groq_fast_model_selected():
    s = Settings(_env_file=None, llm_provider="groq", groq_api_key="test-key")
    assert get_llm("fast", s).model_name == s.groq_fast_model


def test_switch_to_gemini():
    s = Settings(_env_file=None, llm_provider="google_genai", google_api_key="test-key")
    llm = get_llm("main", s)
    assert type(llm).__name__ == "ChatGoogleGenerativeAI"
    assert s.gemini_main_model in llm.model


def test_missing_key_gives_clear_error():
    s = Settings(_env_file=None, llm_provider="groq", groq_api_key=None)
    with pytest.raises(MissingAPIKeyError, match="GROQ_API_KEY"):
        get_llm("main", s)
