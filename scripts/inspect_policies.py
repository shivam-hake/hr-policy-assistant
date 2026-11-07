"""List every policy section and its citation id.
Run:  uv run python -m scripts.inspect_policies"""
from collections import Counter

from hr_assistant.config import get_settings
from hr_assistant.policies import load_policies


def main() -> None:
    docs = load_policies(get_settings().policies_dir)
    for d in docs:
        m = d.metadata
        print(f"{m['citation']:<12} {m['title']:<38} {len(d.page_content.split()):>4} words")
    per_policy = Counter(d.metadata["policy_id"] for d in docs)
    print(f"\n{len(docs)} sections in {len(per_policy)} policies: "
          + ", ".join(f"{k} {v}" for k, v in per_policy.items()))


if __name__ == "__main__":
    main()
