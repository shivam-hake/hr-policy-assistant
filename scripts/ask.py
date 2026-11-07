"""Ask the assistant from the terminal.

    uv run python -m scripts.ask "Can I claim my home internet bill?"
    uv run python -m scripts.ask --eval q21 q35 q41     # evaluation questions, with expected answers

Uses the real LLM. The scored evaluation of all 60 questions comes in Phase 7."""
import sys

from hr_assistant.assistant import HRAssistant
from hr_assistant.eval_set import load_eval_set
from hr_assistant.guardrails import GuardrailViolation


def show(assistant: HRAssistant, question: str, expected=None) -> None:
    try:
        r = assistant.ask(question)
    except GuardrailViolation as exc:
        print(f"\nQ: {question}\nREJECTED: {exc}")
        return
    print(f"\nQ: {question}")
    print(f"[{r.status} | {r.category}] {r.answer}")
    if r.disclaimer:
        print(f"Note: {r.disclaimer}")
    print(f"Cited: {', '.join(r.citations) or '-'}   |   Retrieved: {', '.join(s.citation for s in r.sources)}"
          f"   |   {r.latency_s:.1f}s")
    for w in r.warnings:
        print(f"WARNING: {w}")
    if expected is not None:
        print(f"EXPECTED ({expected.expected}): {expected.reference_answer or '-'}"
              f"  {expected.relevant_sections or ''}")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    assistant = HRAssistant()
    if args[0] == "--eval":
        questions = {q.id: q for q in load_eval_set()}
        for qid in args[1:]:
            show(assistant, questions[qid].question, questions[qid])
    else:
        show(assistant, " ".join(args))


if __name__ == "__main__":
    main()
