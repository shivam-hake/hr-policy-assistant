"""How similar is the best-matching section for answerable vs unanswerable questions?
If answerable questions score clearly higher, a minimum score can help the
assistant say "the policies don't cover this" (used in Phase 5).
Run:  uv run python -m scripts.score_gap"""
import statistics

from hr_assistant.eval_set import load_eval_set
from hr_assistant.vectorstore import get_vectorstore


def main() -> None:
    store = get_vectorstore()
    groups: dict[str, list[float]] = {"answer": [], "not_covered": [], "decline": []}
    for q in load_eval_set():
        top = store.similarity_search_with_relevance_scores(q.question, k=1)
        groups[q.expected].append(top[0][1] if top else 0.0)
    print(f"{'expected':<13}{'n':>4}{'min':>8}{'median':>8}{'max':>8}")
    for name, scores in groups.items():
        print(f"{name:<13}{len(scores):>4}{min(scores):>8.3f}{statistics.median(scores):>8.3f}{max(scores):>8.3f}")
    lowest_answerable = min(groups["answer"])
    above = sum(s >= lowest_answerable for s in groups["not_covered"] + groups["decline"])
    print(f"\nLowest score of an answerable question: {lowest_answerable:.3f}")
    print(f"Unanswerable questions scoring at least that high: {above} of "
          f"{len(groups['not_covered']) + len(groups['decline'])}")
    print("If many unanswerable questions overlap with answerable ones, a score threshold alone "
          "can't separate them; the LLM must also judge whether the retrieved text answers the question.")


if __name__ == "__main__":
    main()
