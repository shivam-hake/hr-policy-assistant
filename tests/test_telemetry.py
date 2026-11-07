"""Telemetry through the real graph, with a fake chat model that reports usage."""
import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda

from hr_assistant.answering import PolicyAnswer
from hr_assistant.assistant import HRAssistant
from hr_assistant.guardrails import InputCheck
from hr_assistant.pricing import cost_usd
from hr_assistant.telemetry import ListSink, TelemetryCallback, TelemetryStore


class OneDoc(BaseRetriever):
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        return [Document(page_content="H\nUp to INR 1,000.", metadata={"citation": "WFH-3.2", "heading": "H"})]


def fake_llm(model="openai/gpt-oss-20b"):
    return GenericFakeChatModel(messages=iter([AIMessage(
        content="x", usage_metadata={"input_tokens": 500, "output_tokens": 100, "total_tokens": 600,
                                     "output_token_details": {"reasoning": 60}},
        response_metadata={"model_name": model})]))


def make(store=None, fail=False):
    def classify(_):
        fake_llm().invoke("classify")  # a real chat-model call inside the node -> callback fires
        return InputCheck(category="hr_policy")

    def answer(_):
        if fail:
            raise RuntimeError("provider down")
        return PolicyAnswer(answerable=True, answer="INR 1,000.", citations=["WFH-3.2"])

    return HRAssistant(retriever=OneDoc(), answer_chain=RunnableLambda(answer),
                       classifier=RunnableLambda(classify), telemetry=store)


def test_llm_call_attributed_to_graph_node_with_cost():
    sink = ListSink()
    r = make().ask("Internet?", callbacks=[TelemetryCallback(sink, "r1")], request_id="r1")
    assert r.request_id == "r1"
    (call,) = sink.for_request("r1")
    assert call["step"] == "check_input" and call["model"] == "openai/gpt-oss-20b"
    assert (call["input_tokens"], call["output_tokens"], call["reasoning_tokens"]) == (500, 100, 60)
    assert call["cost_usd"] == pytest.approx(cost_usd("openai/gpt-oss-20b", 500, 100))


def test_store_records_requests_and_summary(tmp_path):
    store = TelemetryStore(tmp_path / "t.db")
    a = make(store)
    a.ask("Internet?")
    a.ask("Internet again?")
    t = store.summary()
    assert t["requests"] == 2 and t["by_status"] == {"answered": 2} and t["llm_calls"] == 2
    assert t["steps"][0]["step"] == "check_input"


def test_failed_question_recorded_as_error(tmp_path):
    store = TelemetryStore(tmp_path / "t.db")
    with pytest.raises(RuntimeError):
        make(store, fail=True).ask("Internet?")
    assert store.requests()[0]["status"] == "error"


def test_broken_sink_does_not_break_answering():
    def broken(**row):
        raise OSError("disk full")

    r = make().ask("Internet?", callbacks=[TelemetryCallback(broken, "r1")])
    assert r.status == "answered"
