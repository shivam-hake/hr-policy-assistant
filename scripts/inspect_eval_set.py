"""Summary of the evaluation set.
Run:  uv run python -m scripts.inspect_eval_set"""
from collections import Counter

from hr_assistant.eval_set import load_eval_set


def main() -> None:
    qs = load_eval_set()
    print(f"{len(qs)} questions\n")
    print("By category:")
    for cat, n in Counter(q.category for q in qs).most_common():
        print(f"  {cat:<15}{n:>3}")
    print("\nBy expected behaviour:")
    for exp, n in Counter(q.expected for q in qs).most_common():
        print(f"  {exp:<15}{n:>3}")
    print("\nAnswerable questions per policy:")
    per = Counter(s.split("-")[0] for q in qs for s in q.relevant_sections)
    for pol, n in per.most_common():
        print(f"  {pol:<15}{n:>3}")


if __name__ == "__main__":
    main()
