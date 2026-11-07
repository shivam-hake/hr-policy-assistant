import re
from pathlib import Path

import pytest

from hr_assistant.policies import PolicyFormatError, load_policies, parse_policy

POLICIES = Path(__file__).resolve().parent.parent / "data" / "policies"
DOCS = load_policies(POLICIES)
BY_ID = {d.metadata["citation"]: d for d in DOCS}
HEAD = "---\npolicy_id: X\ntitle: X Policy\nversion: 1\neffective: 2026-01-01\n---\n"


def test_all_policies_load_with_unique_citations():
    assert len({d.metadata["policy_id"] for d in DOCS}) == 7
    assert len(BY_ID) == len(DOCS)


def test_known_section_text_and_metadata():
    d = BY_ID["LEAVE-3.2"]
    assert "30 days" in d.page_content
    assert d.metadata["title"] == "Carry forward"
    assert d.metadata["heading"] == "Leave Policy > 3.2 Carry forward"
    assert d.metadata["source"] == "leave.md"


def test_cross_references_point_to_real_sections():
    """Policies refer to each other (e.g. 'see EXIT-4.2'); those ids must exist."""
    topics = {f"{d.metadata['policy_id']}-{d.metadata['section'].split('.')[0]}" for d in DOCS}
    for d in DOCS:
        for ref in re.findall(r"\(([A-Z]+-\d+(?:\.\d+)?)\)", d.page_content):
            assert ref in BY_ID or ref in topics, f"{d.metadata['citation']} refers to missing {ref}"


def test_no_section_is_too_long_for_one_chunk():
    assert max(len(d.page_content.split()) for d in DOCS) < 120


def test_text_directly_under_a_topic_becomes_its_own_section():
    docs = parse_policy(HEAD + "## 1. Intro\nHello.\n### 1.1 Rule\nDo it.\n", "x.md")
    assert [d.metadata["citation"] for d in docs] == ["X-1", "X-1.1"]


@pytest.mark.parametrize("raw, message", [
    ("## 1. Intro\ntext", "front matter"),
    ("---\npolicy_id: X\n---\n## 1. A\ntext", "missing"),
    (HEAD + "## 1. A\n### 2.1 B\ntext", "not under topic"),
    (HEAD + "## Intro\ntext", "must start with a number"),
])
def test_format_errors(raw, message):
    with pytest.raises(PolicyFormatError, match=message):
        parse_policy(raw, "x.md")
