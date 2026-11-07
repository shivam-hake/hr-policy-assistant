"""Checks on the evaluation set itself, so a bad edit fails CI before it
produces misleading scores."""
from collections import Counter
from pathlib import Path

from hr_assistant.eval_set import load_eval_set
from hr_assistant.policies import load_policies

QUESTIONS = load_eval_set()
CITATIONS = {d.metadata["citation"] for d in load_policies(Path(__file__).resolve().parent.parent / "data" / "policies")}


def test_ids_unique():
    assert len({q.id for q in QUESTIONS}) == len(QUESTIONS)


def test_size():
    assert len(QUESTIONS) >= 50


def test_relevant_sections_exist():
    for q in QUESTIONS:
        for s in q.relevant_sections:
            assert s in CITATIONS, f"{q.id} cites missing section {s}"


def test_every_category_has_at_least_three_questions():
    counts = Counter(q.category for q in QUESTIONS)
    assert all(n >= 3 for n in counts.values()), counts
    assert len(counts) == 9


def test_every_policy_is_tested():
    tested = {s.split("-")[0] for q in QUESTIONS for s in q.relevant_sections}
    assert tested == {c.split("-")[0] for c in CITATIONS}


def test_multi_section_questions_need_two_sections():
    assert all(len(q.relevant_sections) >= 2 for q in QUESTIONS if q.category == "multi_section")


def test_mix_of_behaviours():
    counts = Counter(q.expected for q in QUESTIONS)
    assert counts["not_covered"] >= 5 and counts["decline"] >= 8
