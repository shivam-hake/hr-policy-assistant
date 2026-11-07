"""Pipeline tests: real LangGraph graph, fake retriever and fake answer chain."""
import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda

from hr_assistant.answering import NOT_COVERED, PolicyAnswer, verify_citations
from hr_assistant.assistant import HRAssistant
from hr_assistant.guardrails import InputCheck

POLICY_Q = RunnableLambda(lambda _: InputCheck(category="hr_policy"))


class FixedRetriever(BaseRetriever):
    docs: list[Document]

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        return self.docs


def doc(cid, heading, text):
    return Document(page_content=f"{heading}\n{text}", metadata={"citation": cid, "heading": heading})


DOCS = [
    doc("WFH-3.2", "Work from Home > 3.2 Internet reimbursement", "Up to INR 1,000 per month."),
    doc("EXP-2.2", "Expense Policy > 2.2 Home internet", "Not reimbursed under this policy."),
]


def assistant_returning(answer: PolicyAnswer, seen: list | None = None):
    def chain(inputs):
        if seen is not None:
            seen.append(inputs)
        return answer
    return HRAssistant(retriever=FixedRetriever(docs=DOCS), answer_chain=RunnableLambda(chain), classifier=POLICY_Q)


def test_answer_with_valid_citation():
    seen = []
    a = assistant_returning(PolicyAnswer(answerable=True, answer="Up to INR 1,000 a month.",
                                         citations=["WFH-3.2"]), seen)
    r = a.ask("Can I claim my home internet bill?")
    assert r.status == "answered" and r.citations == ["WFH-3.2"]
    assert [s.citation for s in r.cited_sources] == ["WFH-3.2"]
    assert "[WFH-3.2] Work from Home > 3.2 Internet reimbursement" in seen[0]["context"]


def test_not_answerable_gives_standard_message():
    r = assistant_returning(PolicyAnswer(answerable=False, answer="", citations=[])).ask("Relocation?")
    assert r.status == "not_covered" and r.answer == NOT_COVERED and r.citations == []


def test_invented_citation_is_removed_and_answer_withheld():
    r = assistant_returning(PolicyAnswer(answerable=True, answer="Yes, INR 5,000.",
                                         citations=["TRAVEL-9.9"])).ask("?")
    assert r.status == "not_covered"
    assert any("TRAVEL-9.9" in w for w in r.warnings)


def test_partly_invented_citations_keep_the_valid_ones():
    v = verify_citations(PolicyAnswer(answerable=True, answer="x", citations=["WFH-3.2", "FAKE-1", "WFH-3.2"]),
                         ["WFH-3.2", "EXP-2.2"])
    assert v.status == "answered" and v.citations == ["WFH-3.2"] and len(v.warnings) == 1


def test_answerable_but_empty_answer_is_not_covered():
    v = verify_citations(PolicyAnswer(answerable=True, answer="  ", citations=["WFH-3.2"]), ["WFH-3.2"])
    assert v.status == "not_covered"


def test_retry_on_invalid_output():
    """with_retry: a chain that fails twice then succeeds still returns an answer."""
    from hr_assistant.answering import ANSWER_PROMPT  # noqa: F401  (prompt must import cleanly)

    calls = {"n": 0}

    def flaky(_):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("invalid JSON from model")
        return PolicyAnswer(answerable=True, answer="ok", citations=["WFH-3.2"])

    chain = RunnableLambda(flaky).with_retry(stop_after_attempt=3, wait_exponential_jitter=False)
    r = HRAssistant(retriever=FixedRetriever(docs=DOCS), answer_chain=chain, classifier=POLICY_Q).ask("?")
    assert r.status == "answered" and calls["n"] == 3


def test_prompt_renders_with_context_and_question():
    from hr_assistant.answering import ANSWER_PROMPT, format_context

    msgs = ANSWER_PROMPT.invoke({"question": "Q?", "context": format_context(DOCS)}).to_messages()
    assert "[EXP-2.2]" in msgs[0].content and msgs[1].content == "Q?"


@pytest.mark.parametrize("kind", ["answered", "not_covered"])
def test_result_always_has_sources_listed(kind):
    ans = PolicyAnswer(answerable=kind == "answered", answer="x" if kind == "answered" else "",
                       citations=["WFH-3.2"] if kind == "answered" else [])
    r = assistant_returning(ans).ask("?")
    assert len(r.sources) == 2  # retrieved sections are always returned, for transparency
