"""Ingestion tests with LangChain's DeterministicFakeEmbedding: same Chroma
code path as production, but no model download. Retrieval QUALITY is measured
by the evaluation in Phase 4, not here."""
from pathlib import Path

import numpy as np
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from hr_assistant.policies import load_policies
from hr_assistant.vectorstore import build_index, fingerprint, index_status, to_index_document

POLICIES = Path(__file__).resolve().parent.parent / "data" / "policies"
DOCS = load_policies(POLICIES)
EMB = DeterministicFakeEmbedding(size=32)


def build(tmp_path, docs=DOCS):
    return build_index(docs, EMB, str(tmp_path), "hr_policies", "fake-model")


def test_index_document_has_heading_and_stable_id():
    d = to_index_document(next(d for d in DOCS if d.metadata["citation"] == "LEAVE-3.2"))
    assert d.id == "LEAVE-3.2"
    assert d.page_content.startswith("Leave Policy > 3.2 Carry forward\n")


def test_build_indexes_every_section(tmp_path):
    store = build(tmp_path)
    assert store._collection.count() == len(DOCS)
    assert store._collection.metadata["hnsw:space"] == "cosine"


def test_metadata_survives_round_trip(tmp_path):
    store = build(tmp_path)
    hit = store.get(ids=["EXIT-4.2"])
    assert hit["metadatas"][0]["title"] == "Earned leave encashment"
    assert hit["metadatas"][0]["policy_id"] == "EXIT"


def test_rebuild_removes_deleted_sections(tmp_path):
    build(tmp_path)
    store = build(tmp_path, DOCS[:10])  # simulate sections removed from the policies
    assert store._collection.count() == 10


def test_exact_text_is_retrieved_first(tmp_path):
    """Fake embeddings are hash-based: identical text -> identical vector."""
    store = build(tmp_path)
    target = to_index_document(DOCS[5])
    assert store.similarity_search(target.page_content, k=1)[0].id == target.id


def test_index_status_detects_policy_changes(tmp_path):
    assert index_status(DOCS, str(tmp_path), "fake-model") == "missing"
    build(tmp_path)
    assert index_status(DOCS, str(tmp_path), "fake-model") == "ok"
    edited = [d.model_copy() for d in DOCS]
    edited[0].page_content += " (amended)"
    assert index_status(edited, str(tmp_path), "fake-model") == "stale"
    assert index_status(DOCS, str(tmp_path), "another-model") == "stale"


def test_fingerprint_ignores_order():
    assert fingerprint(DOCS, "m") == fingerprint(list(reversed(DOCS)), "m")


def test_fastembed_adapter_uses_query_and_passage_methods(monkeypatch):
    """The adapter must call passage_embed for documents and query_embed for questions."""
    import fastembed

    calls = []

    class FakeTextEmbedding:
        def __init__(self, model_name, cache_dir=None):
            pass

        def passage_embed(self, texts, batch_size=32):
            calls.append("passage")
            return (np.ones(3) for _ in texts)

        def query_embed(self, text):
            calls.append("query")
            return iter([np.zeros(3)])

    monkeypatch.setattr(fastembed, "TextEmbedding", FakeTextEmbedding)
    from hr_assistant.embeddings import FastEmbedEmbeddings

    emb = FastEmbedEmbeddings("any-model")
    assert emb.embed_documents(["a", "b"]) == [[1.0, 1.0, 1.0]] * 2
    assert emb.embed_query("q") == [0.0, 0.0, 0.0]
    assert calls == ["passage", "query"]


def test_missing_index_gives_clear_error(tmp_path, monkeypatch):
    from hr_assistant import vectorstore
    from hr_assistant.config import Settings

    monkeypatch.setattr(vectorstore, "get_settings",
                        lambda: Settings(_env_file=None, chroma_dir=str(tmp_path), policies_dir=str(POLICIES)))
    with pytest.raises(vectorstore.IndexNotReadyError, match="scripts.ingest"):
        vectorstore.get_vectorstore(embeddings=EMB)
