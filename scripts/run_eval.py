"""Full evaluation: all questions through the assistant, scored.

    uv run python -m scripts.run_eval                    # all 60 questions + judge
    uv run python -m scripts.run_eval --ids q21 q35 q41  # a subset
    uv run python -m scripts.run_eval --no-judge         # behaviour and citations only (cheaper)
    uv run python -m scripts.run_eval --min-behaviour 0.85 --max-unsafe 0.05 --min-correct 0.8

Exit code 1 if a threshold is missed (used by CI). Report: eval/results/answers.json.
About 3 LLM calls per question (classifier, answer, judge); expect a few minutes on the free tier."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from hr_assistant.answer_eval import build_judge, run_answer_eval
from hr_assistant.assistant import HRAssistant
from hr_assistant.config import get_settings
from hr_assistant.eval_set import load_eval_set
from hr_assistant.llm import describe
from hr_assistant.telemetry import ListSink, step_stats

OUT = Path("eval/results")


def fmt(v) -> str:
    return "  -  " if v is None else f"{v:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--delay", type=float, default=2.0)
    ap.add_argument("--min-behaviour", type=float, default=0.0)
    ap.add_argument("--max-unsafe", type=float, default=1.0)
    ap.add_argument("--min-correct", type=float, default=0.0)
    args = ap.parse_args()

    questions = load_eval_set()
    if args.ids:
        questions = [q for q in questions if q.id in args.ids]
    s = get_settings()
    assistant = HRAssistant()
    judge = None if args.no_judge else build_judge()
    sink = ListSink()

    print(f"Model: {describe()} | retriever: {s.retriever_kind} k={s.retrieval_k} | "
          f"judge: {'off' if judge is None else 'on'} | questions: {len(questions)}\n")
    print(f"{'id':<5}{'category':<15}{'expected':<12}{'got':<12}{'ok':<4}{'facts':>7}{'faith':>7}{'cite':>7}{'sec':>6}")

    def progress(x):
        print(f"{x.id:<5}{x.category:<15}{x.expected:<12}{x.status:<12}{'Y' if x.behaviour_ok else 'N':<4}"
              f"{fmt(x.fact_coverage):>7}{'' if x.faithful is None else ('Y' if x.faithful else 'N'):>7}"
              f"{fmt(x.citation_recall):>7}{x.latency_s:>6.1f}")

    summary, scores = run_answer_eval(
        questions, lambda q, cb, rid: assistant.ask(q, callbacks=cb, request_id=rid), judge, sink,
        args.delay, progress,
    )

    print("\nPROBLEMS")
    for x in scores:
        if x.error:
            print(f"  {x.id}: ERROR {x.error}")
        elif not x.behaviour_ok:
            print(f"  {x.id}: expected {x.expected}, got {x.status}: {x.answer[:100]}")
        elif x.fully_correct is False or x.faithful is False:
            print(f"  {x.id}: facts {x.fact_coverage}, faithful {x.faithful}: {x.answer[:100]}")
            for u in x.unsupported:
                print(f"        unsupported: {u}")

    print("\nSUMMARY")
    for key in ["behaviour_accuracy", "false_refusal_rate", "unsafe_answer_rate", "fact_coverage",
                "fully_correct_rate", "faithfulness", "citation_recall", "citation_precision", "disclaimer_rate"]:
        print(f"  {key:<22}{fmt(summary[key])}")
    print(f"  {'avg_latency_s':<22}{fmt(summary['avg_latency_s'])}")
    cost = summary["avg_cost_usd"]
    print(f"  {'avg_cost_usd':<22}{'n/a' if cost is None else f'{cost:.6f}'}  (assistant only, judge excluded)")
    print("\nBY CATEGORY (behaviour / fully correct)")
    for cat, m in summary["by_category"].items():
        print(f"  {cat:<15}n={m['n']:<3}{fmt(m['behaviour'])}  {fmt(m['fully_correct'])}")
    print("\nLLM STEPS (assistant)")
    for st in step_stats(sink.calls):
        print(f"  {st['step']:<12} calls {st['calls']:<4} avg {st['avg_latency_ms']} ms  p95 {st['p95_latency_ms']} ms  "
              f"avg tokens {st['avg_input_tokens']} in / {st['avg_output_tokens']} out "
              f"({st['avg_reasoning_tokens']} reasoning)")

    OUT.mkdir(parents=True, exist_ok=True)
    report = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": describe(),
              "retriever": s.retriever_kind, "k": s.retrieval_k, "judge": judge is not None,
              "summary": summary, "questions": [x.model_dump() for x in scores]}
    (OUT / "answers.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    failed = []
    if (summary["behaviour_accuracy"] or 0) < args.min_behaviour:
        failed.append(f"behaviour {summary['behaviour_accuracy']} < {args.min_behaviour}")
    if (summary["unsafe_answer_rate"] or 0) > args.max_unsafe:
        failed.append(f"unsafe answers {summary['unsafe_answer_rate']} > {args.max_unsafe}")
    if judge is not None and (summary["fully_correct_rate"] or 0) < args.min_correct:
        failed.append(f"fully correct {summary['fully_correct_rate']} < {args.min_correct}")
    if failed:
        print("\nQUALITY GATE FAILED: " + "; ".join(failed))
        return 1
    print("\nQuality gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
