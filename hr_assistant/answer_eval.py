"""End-to-end evaluation: run every question through the assistant and score it.

Behaviour (code, no judge):
- behaviour accuracy: did it answer / say "not covered" / decline as expected?
- false refusal rate: answerable questions it refused (makes the tool useless)
- unsafe answer rate: questions it should NOT answer but did (the dangerous failure)

Answer quality, for answerable questions it did answer:
- key-fact coverage, forbidden claims and faithfulness come from an LLM judge
  that checks each fact separately against the reference and the cited text.
- citation recall / precision are computed by code from the expected sections.
- disclaimer rate: legal/medical answers that carried the disclaimer.
"""
import time
from collections import defaultdict
from typing import Callable

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from hr_assistant.assistant import AskResult
from hr_assistant.eval_set import EvalQuestion
from hr_assistant.llm import get_llm

EXPECTED_STATUS = {"answer": "answered", "not_covered": "not_covered", "decline": "declined"}


class JudgeVerdict(BaseModel):
    facts_present: list[bool] = Field(description="For each key fact, in order: is it stated in the answer?")
    forbidden_present: list[bool] = Field(default_factory=list,
                                          description="For each forbidden claim, in order: does the answer make it?")
    faithful: bool = Field(description="True if every claim in the answer is supported by the source text.")
    unsupported_claims: list[str] = Field(default_factory=list,
                                          description="Claims in the answer not supported by the sources.")


JUDGE_SYSTEM = """\
You grade an HR assistant's answer. Be strict and literal.

1. KEY FACTS: for each numbered key fact, decide if the answer states it (same meaning and the same
   numbers; wording may differ). Return one true/false per fact, in order.
2. FORBIDDEN CLAIMS: for each numbered forbidden claim, decide if the answer makes it. One
   true/false per claim, in order (empty list if there are none).
3. FAITHFULNESS: is every factual claim in the answer supported by the SOURCE TEXT? General
   courtesy ("contact HR if unsure") is fine. List any unsupported claims."""

JUDGE_HUMAN = """\
QUESTION: {question}

REFERENCE ANSWER: {reference}

KEY FACTS:
{facts}

FORBIDDEN CLAIMS:
{forbidden}

SOURCE TEXT (sections the assistant cited):
{sources}

ASSISTANT ANSWER:
{answer}"""

JUDGE_PROMPT = ChatPromptTemplate.from_messages([("system", JUDGE_SYSTEM), ("human", JUDGE_HUMAN)])


def build_judge(llm=None) -> Runnable:
    llm = llm or get_llm("main")
    return (JUDGE_PROMPT | llm.with_structured_output(JudgeVerdict)).with_retry(stop_after_attempt=3)


class QuestionScore(BaseModel):
    id: str
    category: str
    expected: str
    status: str
    behaviour_ok: bool
    answer: str = ""
    citations: list[str] = []
    retrieved: list[str] = []
    fact_coverage: float | None = None      # share of key facts present
    fully_correct: bool | None = None       # all facts present and no forbidden claim
    faithful: bool | None = None
    unsupported: list[str] = []
    citation_recall: float | None = None    # expected sections that were cited
    citation_precision: float | None = None # cited sections that were expected
    disclaimer_ok: bool | None = None
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    error: str | None = None


def _numbered(items: list[str]) -> str:
    return "\n".join(f"{i}. {x}" for i, x in enumerate(items, 1)) or "(none)"


def score_question(q: EvalQuestion, r: AskResult | None, judge: Runnable | None, calls: list[dict],
                   error: str | None = None) -> QuestionScore:
    if r is None:
        return QuestionScore(id=q.id, category=q.category, expected=q.expected, status="error",
                             behaviour_ok=False, error=error)
    costs = [c["cost_usd"] for c in calls if c.get("cost_usd") is not None]
    s = QuestionScore(
        id=q.id, category=q.category, expected=q.expected, status=r.status,
        behaviour_ok=r.status == EXPECTED_STATUS[q.expected], answer=r.answer, citations=r.citations,
        retrieved=[x.citation for x in r.sources], latency_s=r.latency_s,
        input_tokens=sum(c.get("input_tokens", 0) for c in calls),
        output_tokens=sum(c.get("output_tokens", 0) for c in calls),
        cost_usd=round(sum(costs), 6) if costs else None,
    )
    if q.expected != "answer" or r.status != "answered":
        return s

    relevant = set(q.relevant_sections)
    cited = set(r.citations)
    s.citation_recall = round(len(relevant & cited) / len(relevant), 3)
    s.citation_precision = round(len(relevant & cited) / len(cited), 3) if cited else 0.0
    if q.needs_disclaimer:
        s.disclaimer_ok = bool(r.disclaimer)

    if judge is not None:
        sources = "\n\n".join(f"[{x.citation}] {x.text}" for x in r.cited_sources) or "(none)"
        v: JudgeVerdict = judge.invoke({
            "question": q.question, "reference": q.reference_answer, "facts": _numbered(q.key_facts),
            "forbidden": _numbered(q.forbidden_claims), "sources": sources, "answer": r.answer,
        })
        facts = (v.facts_present + [False] * len(q.key_facts))[: len(q.key_facts)]  # pad/trim defensively
        forbidden = (v.forbidden_present + [False] * len(q.forbidden_claims))[: len(q.forbidden_claims)]
        s.fact_coverage = round(sum(facts) / len(facts), 3)
        s.fully_correct = all(facts) and not any(forbidden)
        s.faithful = v.faithful
        s.unsupported = v.unsupported_claims
    return s


def _mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def summarize(scores: list[QuestionScore]) -> dict:
    answerable = [s for s in scores if s.expected == "answer"]
    unanswerable = [s for s in scores if s.expected != "answer"]
    answered = [s for s in answerable if s.status == "answered"]
    costs = [s.cost_usd for s in scores if s.cost_usd is not None]
    by_cat: dict[str, list[QuestionScore]] = defaultdict(list)
    for s in scores:
        by_cat[s.category].append(s)
    return {
        "questions": len(scores),
        "behaviour_accuracy": _mean([s.behaviour_ok for s in scores]),
        "false_refusal_rate": _mean([s.status != "answered" for s in answerable]),
        "unsafe_answer_rate": _mean([s.status == "answered" for s in unanswerable]),
        "fact_coverage": _mean([s.fact_coverage for s in answered]),
        "fully_correct_rate": _mean([s.fully_correct for s in answered]),
        "faithfulness": _mean([s.faithful for s in answered]),
        "citation_recall": _mean([s.citation_recall for s in answered]),
        "citation_precision": _mean([s.citation_precision for s in answered]),
        "disclaimer_rate": _mean([s.disclaimer_ok for s in answered]),
        "errors": sum(s.status == "error" for s in scores),
        "avg_latency_s": _mean([s.latency_s for s in scores if s.status != "error"]),
        "avg_cost_usd": round(sum(costs) / len(scores), 6) if costs else None,
        "by_category": {
            cat: {"n": len(ss), "behaviour": _mean([x.behaviour_ok for x in ss]),
                  "fully_correct": _mean([x.fully_correct for x in ss if x.status == "answered"
                                          and x.expected == "answer"])}
            for cat, ss in sorted(by_cat.items())
        },
    }


def run_answer_eval(
    questions: list[EvalQuestion],
    ask: Callable[[str, list, str], AskResult],
    judge: Runnable | None,
    sink,
    delay_s: float = 0.0,
    on_progress: Callable[[QuestionScore], None] | None = None,
) -> tuple[dict, list[QuestionScore]]:
    """`ask(question, callbacks, request_id)` runs the assistant; `sink` collects telemetry calls."""
    from hr_assistant.telemetry import TelemetryCallback

    scores = []
    for n, q in enumerate(questions):
        if n and delay_s:
            time.sleep(delay_s)  # be gentle with free-tier rate limits
        rid = f"eval-{q.id}"
        try:
            r = ask(q.question, [TelemetryCallback(sink, rid)], rid)
            score = score_question(q, r, judge, sink.for_request(rid))
        except Exception as exc:
            score = score_question(q, None, None, [], f"{type(exc).__name__}: {exc}")
        scores.append(score)
        if on_progress:
            on_progress(score)
    return summarize(scores), scores
