"""Retriever and metric tests. Fake embeddings for the vector side, real BM25."""
from pathlib import Path

from langchain_core.embeddings import DeterministicFakeEmbedding

from hr_assistant.policies import load_policies
from hr_assistant.retrieval_eval import QuestionResult, failures, full_recall_at, hit_at, metrics, reciprocal_rank
from hr_assistant.retrievers import BM25PolicyRetriever, build_retriever, tokenize
from hr_assistant.vectorstore import build_index

POLICIES = Path(__file__).resolve().parent.parent / "data" / "policies"
DOCS = load_policies(POLICIES)


def ids(docs):
    return [d.metadata["citation"] for d in docs]


def test_tokenize_drops_stopwords_and_plural_s():
    assert tokenize("How many casual leaves do I get?") == ["casual", "leave", "get"]


def test_bm25_finds_exact_terms():
    r = BM25PolicyRetriever.from_documents(DOCS, k=3)
    assert ids(r.invoke("per diem international travel"))[0] == "TRAVEL-4.2"
    assert "EXIT-2.1" in ids(r.invoke("notice period length"))


def test_bm25_returns_nothing_for_unknown_words():
    assert BM25PolicyRetriever.from_documents(DOCS).invoke("xylophone quokka") == []


def test_factory_builds_each_kind(tmp_path):
    store = build_index(DOCS, DeterministicFakeEmbedding(size=32), str(tmp_path), "hr_policies", "fake")
    for kind in ("vector", "bm25", "hybrid"):
        docs = build_retriever(kind, store, DOCS, k=3).invoke("casual leave per year")
        assert docs and all("citation" in d.metadata for d in docs)


def test_hybrid_merges_without_duplicates(tmp_path):
    store = build_index(DOCS, DeterministicFakeEmbedding(size=32), str(tmp_path), "hr_policies", "fake")
    got = ids(build_retriever("hybrid", store, DOCS, k=4).invoke("casual leave"))
    assert len(got) == len(set(got))
    assert "LEAVE-2.1" in got  # BM25 side finds it even though fake vectors are random


R = QuestionResult(id="q", category="multi_section", question="?", relevant=["A-1", "B-2"],
                   retrieved=["X-9", "A-1", "Y-3", "B-2"])


def test_metric_definitions():
    assert hit_at(R, 1) is False and hit_at(R, 2) is True
    assert full_recall_at(R, 3) is False and full_recall_at(R, 4) is True
    assert reciprocal_rank(R, 4) == 0.5 and reciprocal_rank(R, 1) == 0.0


def test_metrics_aggregate():
    perfect = R.model_copy(update={"retrieved": ["A-1", "B-2"]})
    m = metrics([R, perfect], k=2)
    assert m == {"k": 2, "questions": 2, "hit": 1.0, "full_recall": 0.5, "mrr": 0.75}
    assert failures([R, perfect], 2) == [R]
