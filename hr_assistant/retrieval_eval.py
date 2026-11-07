"""Retrieval metrics over the answerable evaluation questions.

For each question we know which sections answer it (relevant_sections):
- hit@k:          is at least one relevant section in the top k?
- full_recall@k:  are ALL relevant sections in the top k? (matters for multi-section questions)
- MRR:            1 / rank of the first relevant section (1.0 = always first)

If retrieval misses the section, no prompt can rescue the answer, so this is
measured on its own before any LLM is involved. It needs no API calls.
"""
from collections import defaultdict

from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel

from hr_assistant.eval_set import EvalQuestion


class QuestionResult(BaseModel):
    id: str
    category: str
    question: str
    relevant: list[str]
    retrieved: list[str]     # citations in ranked order


def hit_at(r: QuestionResult, k: int) -> bool:
    return any(c in r.relevant for c in r.retrieved[:k])


def full_recall_at(r: QuestionResult, k: int) -> bool:
    return set(r.relevant) <= set(r.retrieved[:k])


def reciprocal_rank(r: QuestionResult, k: int) -> float:
    for i, c in enumerate(r.retrieved[:k], start=1):
        if c in r.relevant:
            return 1.0 / i
    return 0.0


def run_retrieval(retriever: BaseRetriever, questions: list[EvalQuestion]) -> list[QuestionResult]:
    results = []
    for q in questions:
        if q.expected != "answer":
            continue
        docs = retriever.invoke(q.question)
        results.append(QuestionResult(
            id=q.id, category=q.category, question=q.question,
            relevant=q.relevant_sections, retrieved=[d.metadata["citation"] for d in docs],
        ))
    return results


def metrics(results: list[QuestionResult], k: int) -> dict:
    n = len(results) or 1
    return {
        "k": k,
        "questions": len(results),
        "hit": round(sum(hit_at(r, k) for r in results) / n, 3),
        "full_recall": round(sum(full_recall_at(r, k) for r in results) / n, 3),
        "mrr": round(sum(reciprocal_rank(r, k) for r in results) / n, 3),
    }


def metrics_by_category(results: list[QuestionResult], k: int) -> dict[str, dict]:
    groups: dict[str, list[QuestionResult]] = defaultdict(list)
    for r in results:
        groups[r.category].append(r)
    return {cat: metrics(rs, k) for cat, rs in sorted(groups.items())}


def failures(results: list[QuestionResult], k: int) -> list[QuestionResult]:
    """Questions where not every relevant section made it into the top k."""
    return [r for r in results if not full_recall_at(r, k)]
