"""Evaluation scoring tests: fake assistant results and a fake judge."""
from langchain_core.runnables import RunnableLambda

from hr_assistant.answer_eval import JudgeVerdict, run_answer_eval, score_question, summarize
from hr_assistant.assistant import AskResult, Source
from hr_assistant.eval_set import EvalQuestion
from hr_assistant.telemetry import ListSink

Q_ANSWER = EvalQuestion(id="q1", question="Internet?", category="paraphrase", expected="answer",
                        relevant_sections=["WFH-3.2"], reference_answer="Up to INR 1,000.",
                        key_facts=["INR 1,000 per month", "approved arrangement"])
Q_NOT = EvalQuestion(id="q2", question="Relocation?", category="not_covered", expected="not_covered")
Q_LEGAL = Q_ANSWER.model_copy(update={"id": "q3", "needs_disclaimer": True})


def result(status="answered", citations=("WFH-3.2", "EXP-2.2"), disclaimer=""):
    return AskResult(question="?", status=status, category="hr_policy", answer="Up to INR 1,000 a month.",
                     citations=list(citations),
                     sources=[Source(citation="WFH-3.2", heading="h", text="t"),
                              Source(citation="EXP-2.2", heading="h", text="t")],
                     disclaimer=disclaimer, latency_s=1.0)


JUDGE = RunnableLambda(lambda _: JudgeVerdict(facts_present=[True, False], faithful=True))


def test_answer_scored_for_facts_and_citations():
    s = score_question(Q_ANSWER, result(), JUDGE, [])
    assert s.behaviour_ok and s.fact_coverage == 0.5 and s.fully_correct is False
    assert s.citation_recall == 1.0 and s.citation_precision == 0.5


def test_wrong_behaviour_detected():
    s = score_question(Q_NOT, result(status="answered"), None, [])
    assert s.behaviour_ok is False


def test_judge_list_length_mismatch_is_handled():
    short = RunnableLambda(lambda _: JudgeVerdict(facts_present=[True], faithful=True))
    assert score_question(Q_ANSWER, result(), short, []).fact_coverage == 0.5


def test_disclaimer_checked_only_when_needed():
    assert score_question(Q_LEGAL, result(disclaimer="Not legal advice."), None, []).disclaimer_ok is True
    assert score_question(Q_LEGAL, result(), None, []).disclaimer_ok is False
    assert score_question(Q_ANSWER, result(), None, []).disclaimer_ok is None


def test_summary_rates():
    scores = [
        score_question(Q_ANSWER, result(), JUDGE, []),                    # answered correctly (behaviour)
        score_question(Q_ANSWER, result(status="not_covered"), JUDGE, []),  # false refusal
        score_question(Q_NOT, result(status="answered"), None, []),       # unsafe answer
        score_question(Q_NOT, result(status="not_covered"), None, []),    # correct refusal
    ]
    m = summarize(scores)
    assert m["behaviour_accuracy"] == 0.5
    assert m["false_refusal_rate"] == 0.5 and m["unsafe_answer_rate"] == 0.5


def test_run_collects_cost_and_survives_errors():
    sink = ListSink()

    def ask(question, callbacks, rid):
        if question == "Relocation?":
            raise RuntimeError("rate limited")
        sink(request_id=rid, step="generate", input_tokens=100, output_tokens=10, cost_usd=0.001,
             latency_ms=5, status="ok")
        return result()

    summary, scores = run_answer_eval([Q_ANSWER, Q_NOT], ask, JUDGE, sink)
    assert scores[0].cost_usd == 0.001 and scores[0].input_tokens == 100
    assert scores[1].status == "error" and summary["errors"] == 1
