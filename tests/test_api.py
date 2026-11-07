"""API tests: real FastAPI app, fake assistant components."""
import pytest
from fastapi.testclient import TestClient
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda

from hr_assistant.answering import PolicyAnswer
from hr_assistant.api import app
from hr_assistant.assistant import HRAssistant
from hr_assistant.guardrails import InputCheck
from hr_assistant.telemetry import TelemetryStore


class OneDoc(BaseRetriever):
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        return [Document(page_content="H\nUp to INR 1,000 per month.",
                         metadata={"citation": "WFH-3.2", "heading": "Work from Home > 3.2 Internet"})]


@pytest.fixture
def client(tmp_path):
    def make(category="hr_policy", fail=False):
        def answer(_):
            if fail:
                raise RuntimeError("provider down")
            return PolicyAnswer(answerable=True, answer="Up to INR 1,000 a month.", citations=["WFH-3.2"])

        app.state.telemetry = TelemetryStore(tmp_path / "t.db")
        app.state.assistant = HRAssistant(
            retriever=OneDoc(), answer_chain=RunnableLambda(answer),
            classifier=RunnableLambda(lambda _: InputCheck(category=category)), telemetry=app.state.telemetry)
        return TestClient(app)

    yield make
    del app.state.assistant
    del app.state.telemetry


def test_ask_returns_answer_and_cited_sources(client):
    r = client().post("/ask", json={"question": "Can I claim my home internet bill?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answered" and body["citations"] == ["WFH-3.2"]
    assert body["cited_sources"][0]["heading"] == "Work from Home > 3.2 Internet"


def test_declined_question(client):
    body = client(category="off_topic").post("/ask", json={"question": "Capital of Australia?"}).json()
    assert body["status"] == "declined" and body["sources"] == []


@pytest.mark.parametrize("q", ["", "   ", "x" * 1500])
def test_invalid_question_422(client, q):
    assert client().post("/ask", json={"question": q}).status_code == 422


def test_llm_failure_502(client):
    assert client(fail=True).post("/ask", json={"question": "Internet?"}).status_code == 502


def test_policies_endpoints(client):
    c = client()
    sections = c.get("/policies").json()
    assert len(sections) == 74 and {"citation", "heading", "policy", "text"} <= sections[0].keys()
    assert "30 days" in c.get("/policies/leave-3.2").json()["text"]
    assert c.get("/policies/NOPE-1").status_code == 404


def test_telemetry_endpoint(client):
    c = client()
    c.post("/ask", json={"question": "Internet?"})
    t = c.get("/telemetry").json()
    assert t["summary"]["requests"] == 1 and t["recent"][0]["status"] == "answered"
