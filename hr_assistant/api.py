"""REST API for the HR Policy Assistant.

Run locally:  uv run uvicorn hr_assistant.api:app --reload
Docs:         http://localhost:8000/docs

On startup the policy index is checked and rebuilt if missing or outdated, so
a fresh container (or an edited policy) never serves stale answers.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from hr_assistant.assistant import HRAssistant
from hr_assistant.config import get_settings
from hr_assistant.guardrails import MAX_QUESTION_CHARS, GuardrailViolation
from hr_assistant.llm import MissingAPIKeyError
from hr_assistant.policies import load_policies
from hr_assistant.telemetry import TelemetryStore

log = logging.getLogger(__name__)


def ensure_index() -> str:
    from hr_assistant.vectorstore import index_status, ingest

    s = get_settings()
    status = index_status(load_policies(s.policies_dir), s.chroma_dir, s.embedding_model)
    if status != "ok":
        log.warning("Policy index is %s; rebuilding.", status)
        ingest()
    return "ok"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "assistant"):  # tests inject their own
        ensure_index()
        app.state.telemetry = TelemetryStore(get_settings().telemetry_db_path)
        app.state.assistant = HRAssistant(telemetry=app.state.telemetry)
    yield


app = FastAPI(
    title="HR Policy Assistant",
    version="1.0.0",
    description="Answers employee questions from Kestrel's HR policies, with citations.",
    lifespan=lifespan,
)


class AskIn(BaseModel):
    question: str = Field(max_length=MAX_QUESTION_CHARS * 2, examples=["Can I claim my home internet bill?"])


@app.get("/health", tags=["system"])
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", tags=["assistant"])
def ask(body: AskIn, request: Request) -> dict:
    """Answer a question. `status` is answered, not_covered or declined."""
    try:
        r = request.app.state.assistant.ask(body.question)
    except GuardrailViolation as exc:
        raise HTTPException(422, str(exc)) from exc
    except MissingAPIKeyError as exc:
        raise HTTPException(500, str(exc)) from exc
    except Exception as exc:  # LLM provider errors after retries
        log.exception("Answering failed")
        raise HTTPException(502, "The language model is unavailable right now. Please try again.") from exc
    return {**r.model_dump(), "cited_sources": [s.model_dump() for s in r.cited_sources]}


@app.get("/policies", tags=["policies"])
def policies() -> list[dict]:
    """All policy sections with their text, for browsing."""
    return [{"citation": d.metadata["citation"], "heading": d.metadata["heading"],
             "policy": d.metadata["policy_title"], "text": d.page_content}
            for d in load_policies(get_settings().policies_dir)]


@app.get("/policies/{citation}", tags=["policies"])
def policy_section(citation: str) -> dict:
    for d in load_policies(get_settings().policies_dir):
        if d.metadata["citation"].upper() == citation.upper():
            return {"citation": d.metadata["citation"], "heading": d.metadata["heading"],
                    "text": d.page_content, "version": d.metadata["version"],
                    "effective": d.metadata["effective"]}
    raise HTTPException(404, f"Section {citation!r} not found.")


@app.get("/telemetry", tags=["telemetry"])
def telemetry(request: Request) -> dict:
    store: TelemetryStore | None = getattr(request.app.state, "telemetry", None)
    if store is None:
        return {"summary": None, "recent": []}
    return {"summary": store.summary(), "recent": store.requests(limit=20)}
