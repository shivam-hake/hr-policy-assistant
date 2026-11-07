"""Compare retrievers (vector, bm25, hybrid) at several k on the evaluation set.
Run:  uv run python -m scripts.eval_retrieval
      uv run python -m scripts.eval_retrieval --kinds vector hybrid --ks 3 4 5 --show-failures hybrid:4
No LLM calls: retrieval only. Results saved to eval/results/retrieval.json."""
import argparse
import json
from pathlib import Path

from hr_assistant.config import get_settings
from hr_assistant.eval_set import load_eval_set
from hr_assistant.policies import load_policies
from hr_assistant.retrieval_eval import failures, metrics, metrics_by_category, run_retrieval
from hr_assistant.retrievers import build_retriever
from hr_assistant.vectorstore import get_vectorstore

OUT = Path("eval/results")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["vector", "bm25", "hybrid"])
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 3, 4, 5])
    ap.add_argument("--show-failures", default=None, help="e.g. hybrid:4")
    args = ap.parse_args()

    s = get_settings()
    questions = load_eval_set()
    store = get_vectorstore()
    docs = load_policies(s.policies_dir)
    max_k = max(args.ks)

    report = {}
    all_results = {}
    print(f"{'retriever':<10}{'k':>3}{'hit':>8}{'full rec':>10}{'MRR':>8}")
    for kind in args.kinds:
        # Retrieve once at the largest k; smaller k are prefixes of the same ranking.
        results = run_retrieval(build_retriever(kind, store, docs, max_k, s.hybrid_weights), questions)
        all_results[kind] = results
        for k in args.ks:
            m = metrics(results, k)
            report[f"{kind}@{k}"] = {**m, "by_category": metrics_by_category(results, k)}
            print(f"{kind:<10}{k:>3}{m['hit']:>8.3f}{m['full_recall']:>10.3f}{m['mrr']:>8.3f}")
        print()

    k_detail = s.retrieval_k if s.retrieval_k in args.ks else max_k
    print(f"Full recall by category at k={k_detail}:")
    cats = sorted(report[f"{args.kinds[0]}@{k_detail}"]["by_category"])
    print(f"{'category':<15}" + "".join(f"{kind:>9}" for kind in args.kinds))
    for cat in cats:
        row = "".join(f"{report[f'{kind}@{k_detail}']['by_category'][cat]['full_recall']:>9.3f}" for kind in args.kinds)
        print(f"{cat:<15}{row}")

    if args.show_failures:
        kind, k = args.show_failures.split(":")
        print(f"\nMisses for {kind} at k={k}:")
        for r in failures(all_results[kind], int(k)):
            print(f"  {r.id} [{r.category}] {r.question}")
            print(f"      needed {r.relevant}, got {r.retrieved[:int(k)]}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "retrieval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {OUT / 'retrieval.json'}")


if __name__ == "__main__":
    main()
