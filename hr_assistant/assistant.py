"""The question-answering pipeline as a LangGraph graph.

    START -> check_input --(hr_policy)--> retrieve -> generate -> verify -> add_disclaimer -> END
                     \\--(off_topic / injection / personal_data)--> decline -> END

Each step is a node that reads and writes a shared state. Extending the
pipeline means adding a node and an edge; the other nodes don't change.
"""
import time
import uuid
from typing import TypedDict

from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from hr_assistant.answering import PolicyAnswer, build_answer_chain, verify_citations
from hr_assistant.guardrails import (
    DECLINE_MESSAGES,
    DISCLAIMERS,
    build_input_classifier,
    check_input,
    validate_question,
)
from hr_assistant.telemetry import TelemetryCallback, TelemetryStore


class Source(BaseModel):
    citation: str
    heading: str
    text: str


class AssistantState(TypedDict, total=False):
    question: str
    category: str                # hr_policy | off_topic | injection | personal_data
    sensitivity: str             # none | legal | medical
    sources: list[dict]          # retrieved sections (citation, heading, text), in rank order
    raw: dict                    # the model's PolicyAnswer, kept for debugging and evaluation
    status: str                  # answered | not_covered | declined
    answer: str
    citations: list[str]
    disclaimer: str
    warnings: list[str]


class AskResult(BaseModel):
    request_id: str = ""
    question: str
    status: str
    category: str
    answer: str
    citations: list[str]
    sources: list[Source]
    disclaimer: str = ""
    warnings: list[str] = []
    latency_s: float = 0.0

    @property
    def cited_sources(self) -> list[Source]:
        return [s for s in self.sources if s.citation in self.citations]


def build_graph(retriever: BaseRetriever, answer_chain: Runnable, classifier: Runnable):
    def check(state: AssistantState) -> dict:
        result = check_input(state["question"], classifier)
        return {"category": result.category, "sensitivity": result.sensitivity,
                "sources": [], "citations": [], "warnings": [], "disclaimer": ""}

    def route(state: AssistantState) -> str:
        return "retrieve" if state["category"] == "hr_policy" else "decline"

    def decline(state: AssistantState) -> dict:
        return {"status": "declined", "answer": DECLINE_MESSAGES[state["category"]]}

    def retrieve(state: AssistantState) -> dict:
        docs = retriever.invoke(state["question"])
        return {"sources": [
            {"citation": d.metadata["citation"], "heading": d.metadata["heading"],
             "text": d.page_content.split("\n", 1)[-1]}
            for d in docs
        ]}

    def generate(state: AssistantState) -> dict:
        context = "\n\n".join(f"[{s['citation']}] {s['heading']}\n{s['text']}" for s in state["sources"])
        result: PolicyAnswer = answer_chain.invoke({"question": state["question"], "context": context})
        return {"raw": result.model_dump()}

    def verify(state: AssistantState) -> dict:
        v = verify_citations(PolicyAnswer.model_validate(state["raw"]),
                             [s["citation"] for s in state["sources"]])
        return {"status": v.status, "answer": v.answer, "citations": v.citations,
                "warnings": state.get("warnings", []) + v.warnings}

    def add_disclaimer(state: AssistantState) -> dict:
        text = DISCLAIMERS.get(state.get("sensitivity", "none"), "")
        return {"disclaimer": text if state["status"] == "answered" else ""}

    g = StateGraph(AssistantState)
    for name, fn in [("check_input", check), ("decline", decline), ("retrieve", retrieve),
                     ("generate", generate), ("verify", verify), ("add_disclaimer", add_disclaimer)]:
        g.add_node(name, fn)
    g.add_edge(START, "check_input")
    g.add_conditional_edges("check_input", route, ["retrieve", "decline"])
    g.add_edge("decline", END)
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "verify")
    g.add_edge("verify", "add_disclaimer")
    g.add_edge("add_disclaimer", END)
    return g.compile()


class HRAssistant:
    """Entry point used by scripts now and by the API later."""

    def __init__(self, retriever: BaseRetriever | None = None, answer_chain: Runnable | None = None,
                 classifier: Runnable | None = None, telemetry: TelemetryStore | None = None):
        if retriever is None:
            from hr_assistant.retrievers import get_retriever
            retriever = get_retriever()
        self.graph = build_graph(retriever, answer_chain or build_answer_chain(),
                                 classifier or build_input_classifier())
        self.telemetry = telemetry

    def ask(self, question: str, callbacks: list | None = None, request_id: str | None = None) -> AskResult:
        question = validate_question(question)  # raises GuardrailViolation
        request_id = request_id or uuid.uuid4().hex[:12]
        handlers = list(callbacks or [])
        if self.telemetry:
            handlers.append(TelemetryCallback(self.telemetry.add_llm_call, request_id))
        start = time.perf_counter()
        try:
            s = self.graph.invoke({"question": question}, {"callbacks": handlers})
        except Exception as exc:
            self._record(request_id, question, "error", None, start, f"{type(exc).__name__}: {exc}")
            raise
        self._record(request_id, question, s["status"], s["category"], start)
        return AskResult(
            request_id=request_id, question=question, status=s["status"], category=s["category"],
            answer=s["answer"], citations=s.get("citations", []),
            sources=[Source(**x) for x in s.get("sources", [])], disclaimer=s.get("disclaimer", ""),
            warnings=s.get("warnings", []), latency_s=round(time.perf_counter() - start, 3),
        )

    def _record(self, request_id, question, status, category, start, error=None) -> None:
        if not self.telemetry:
            return
        try:
            self.telemetry.add_request(request_id, question, status, category,
                                       int((time.perf_counter() - start) * 1000), error)
        except Exception:  # telemetry must never break answering
            pass
