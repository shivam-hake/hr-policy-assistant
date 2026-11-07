"""Ingestion: policy Documents -> embeddings -> Chroma.

Design choices
- Contextual chunk header: each chunk is stored as
  "Leave Policy > 3.2 Carry forward\\n<rule text>". The heading tells the
  embedding (and later the LLM) what the rule is about, which helps short
  rules like "Up to 30 days..." match questions about carrying leave forward.
- Stable ids: each chunk's id is its citation ("LEAVE-3.2").
- Clean rebuild: ingestion deletes and recreates the collection, so a section
  removed from a policy can never linger in the index.
- Fingerprint: a hash of the policy text + embedding model is saved next to
  the index, so the app can tell when the index is out of date.
"""
import hashlib
import json
from pathlib import Path

from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from hr_assistant.config import get_settings
from hr_assistant.policies import load_policies

MANIFEST = "manifest.json"
_CHROMA_SETTINGS = ChromaSettings(anonymized_telemetry=False)


class IndexNotReadyError(RuntimeError):
    pass


def to_index_document(doc: Document) -> Document:
    return Document(
        page_content=f"{doc.metadata['heading']}\n{doc.page_content}",
        metadata=dict(doc.metadata),
        id=doc.metadata["citation"],
    )


def fingerprint(docs: list[Document], embedding_model: str) -> str:
    h = hashlib.sha256(embedding_model.encode())
    for d in sorted(docs, key=lambda d: d.metadata["citation"]):
        h.update(d.metadata["citation"].encode())
        h.update(d.page_content.encode())
    return h.hexdigest()[:16]


def _open(embeddings: Embeddings, persist_dir: str, collection: str) -> Chroma:
    return Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"},  # BGE vectors are compared by cosine similarity
        client_settings=_CHROMA_SETTINGS,
    )


def build_index(
    docs: list[Document], embeddings: Embeddings, persist_dir: str, collection: str, embedding_model: str
) -> Chroma:
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    store = _open(embeddings, persist_dir, collection)
    store.delete_collection()                      # clean rebuild
    store = _open(embeddings, persist_dir, collection)
    index_docs = [to_index_document(d) for d in docs]
    store.add_documents(index_docs, ids=[d.id for d in index_docs])
    manifest = {"fingerprint": fingerprint(docs, embedding_model), "embedding_model": embedding_model,
                "sections": len(docs)}
    (Path(persist_dir) / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return store


def index_status(docs: list[Document], persist_dir: str, embedding_model: str) -> str:
    """'missing', 'stale' or 'ok'."""
    path = Path(persist_dir) / MANIFEST
    if not path.exists():
        return "missing"
    saved = json.loads(path.read_text(encoding="utf-8"))
    return "ok" if saved.get("fingerprint") == fingerprint(docs, embedding_model) else "stale"


def ingest(embeddings: Embeddings | None = None) -> Chroma:
    """Load the policies and (re)build the index using the configured settings."""
    from hr_assistant.embeddings import get_embeddings

    s = get_settings()
    return build_index(load_policies(s.policies_dir), embeddings or get_embeddings(),
                       s.chroma_dir, s.chroma_collection, s.embedding_model)


def get_vectorstore(embeddings: Embeddings | None = None, require_fresh: bool = True) -> Chroma:
    """Open the existing index. Refuses to run on a missing or outdated index,
    because answers from stale policies would be confidently wrong."""
    from hr_assistant.embeddings import get_embeddings

    s = get_settings()
    if require_fresh:
        status = index_status(load_policies(s.policies_dir), s.chroma_dir, s.embedding_model)
        if status != "ok":
            raise IndexNotReadyError(
                f"The policy index is {status}. Run: uv run python -m scripts.ingest"
            )
    return _open(embeddings or get_embeddings(), s.chroma_dir, s.chroma_collection)
