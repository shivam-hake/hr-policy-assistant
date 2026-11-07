"""Central configuration, loaded from environment variables / .env.
Same pattern as the Meeting Agent: no other file reads os.environ."""
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["groq", "google_genai"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: Provider = "groq"
    groq_api_key: str | None = None
    google_api_key: str | None = None

    groq_main_model: str = "openai/gpt-oss-120b"
    groq_fast_model: str = "openai/gpt-oss-20b"
    gemini_main_model: str = "gemini-2.5-flash"
    gemini_fast_model: str = "gemini-2.5-flash-lite"

    llm_temperature: float = 0.0
    llm_timeout_s: int = 60
    llm_max_retries: int = 2
    groq_reasoning_effort: str | None = "low"

    # RAG settings
    policies_dir: str = "data/policies"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache_dir: str = "data/runtime/models"   # downloaded model files (~130 MB)
    chroma_dir: str = "data/runtime/chroma"
    chroma_collection: str = "hr_policies"
    retrieval_k: int = 4
    # vector | bm25 | hybrid. Chosen from scripts.eval_retrieval results, not by guesswork.
    retriever_kind: str = "vector"
    hybrid_weights: tuple[float, float] = (0.5, 0.5)    # (vector, bm25)

    telemetry_db_path: str = "data/runtime/telemetry.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
