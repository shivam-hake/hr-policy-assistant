"""FastEmbed as a LangChain Embeddings class.

LangChain's own FastEmbed wrapper lives in langchain-community, which is being
sunset. Implementing the two methods of the standard `Embeddings` interface
ourselves takes a few lines, and to the rest of LangChain (Chroma, retrievers)
this class is indistinguishable from an official integration. Swapping to
another embedding model later means writing another class like this one.

FastEmbed runs ONNX models on the CPU: no PyTorch, small Docker image, no API
calls, no rate limits.
"""
from functools import lru_cache
from pathlib import Path

from langchain_core.embeddings import Embeddings

from hr_assistant.config import get_settings


class FastEmbedEmbeddings(Embeddings):
    def __init__(self, model_name: str, cache_dir: str | None = None, batch_size: int = 32):
        from fastembed import TextEmbedding  # imported here so tests can run without the model

        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    # Retrieval models like BGE embed questions and passages slightly
    # differently; FastEmbed's query_embed / passage_embed handle that for us.
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.passage_embed(texts, batch_size=self.batch_size)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


@lru_cache
def get_embeddings() -> Embeddings:
    """One shared model per process: loading it takes a second or two."""
    s = get_settings()
    return FastEmbedEmbeddings(s.embedding_model, cache_dir=s.embedding_cache_dir)
