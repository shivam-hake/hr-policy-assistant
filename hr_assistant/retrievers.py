"""Retrievers: vector, keyword (BM25) and hybrid.

All three are LangChain retrievers (`.invoke(question) -> list[Document]`),
so the rest of the pipeline doesn't care which one is in use; the choice is a
setting (RETRIEVER_KIND) backed by the retrieval evaluation.

- vector: Chroma similarity search. Understands meaning ("pay my phone bill"
  ~ "mobile phone reimbursement") but can blur exact terms.
- bm25: classic keyword ranking. Strong on exact terms ("casual leave",
  "M3", "per diem") but blind to synonyms.
- hybrid: LangChain's EnsembleRetriever merges both rankings with Reciprocal
  Rank Fusion (RRF): a section ranked high by either retriever rises to the top.

BM25 is implemented here as a small BaseRetriever on top of rank_bm25 rather
than taken from the langchain-community package, which is being sunset.
"""
import re

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from pydantic import ConfigDict
from rank_bm25 import BM25Okapi

from hr_assistant.config import get_settings
from hr_assistant.policies import load_policies
from hr_assistant.vectorstore import to_index_document

STOPWORDS = set(
    "a an the to of for and or in on at by with from is are be can do does i my me we our you your "
    "it this that what how when where which who much many any if will would should".split()
)


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w not in STOPWORDS]


class BM25PolicyRetriever(BaseRetriever):
    """Keyword retriever over the same chunks (heading + text) as the vector index."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    docs: list[Document]
    bm25: BM25Okapi
    k: int = 4

    @classmethod
    def from_documents(cls, docs: list[Document], k: int = 4) -> "BM25PolicyRetriever":
        index_docs = [to_index_document(d) for d in docs]
        return cls(docs=index_docs, bm25=BM25Okapi([tokenize(d.page_content) for d in index_docs]), k=k)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)
        return [self.docs[i] for i in ranked[: self.k] if scores[i] > 0]


def build_retriever(
    kind: str,
    vectorstore: VectorStore,
    docs: list[Document],
    k: int,
    weights: tuple[float, float] = (0.5, 0.5),
) -> BaseRetriever:
    vector = vectorstore.as_retriever(search_kwargs={"k": k})
    if kind == "vector":
        return vector
    bm25 = BM25PolicyRetriever.from_documents(docs, k=k)
    if kind == "bm25":
        return bm25
    if kind == "hybrid":
        return EnsembleRetriever(retrievers=[vector, bm25], weights=list(weights), id_key="citation")
    raise ValueError(f"Unknown retriever kind {kind!r}: use vector, bm25 or hybrid")


def get_retriever(kind: str | None = None, k: int | None = None) -> BaseRetriever:
    """The retriever the app uses, built from settings."""
    from hr_assistant.vectorstore import get_vectorstore

    s = get_settings()
    return build_retriever(
        kind or s.retriever_kind, get_vectorstore(), load_policies(s.policies_dir),
        k or s.retrieval_k, s.hybrid_weights,
    )
