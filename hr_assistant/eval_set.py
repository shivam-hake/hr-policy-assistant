"""The evaluation question set: schema and loader.

Each question says what a correct response looks like:
- expected = "answer":      answer from the policies, citing relevant_sections
- expected = "not_covered": HR question the policies don't cover -> say so, don't invent
- expected = "decline":     off-topic, prompt injection or another person's data -> decline

See data/eval/EVAL_GUIDE.md for the rules behind each category.
"""
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator

Expected = Literal["answer", "not_covered", "decline"]
Category = Literal[
    "single",          # one section answers it, wording close to the policy
    "paraphrase",      # one section, but phrased differently or near a look-alike rule
    "conditional",     # answer depends on grade, probation or other conditions
    "multi_section",   # needs two or more sections
    "not_covered",     # HR topic the policies don't mention
    "off_topic",       # not an HR policy question at all
    "injection",       # tries to override the assistant's rules
    "personal_data",   # asks about another person's private data
    "sensitive",       # legal or medical angle: answer plus a disclaimer
]
EVAL_FILE = Path(__file__).resolve().parent.parent / "data" / "eval" / "questions.jsonl"


class EvalQuestion(BaseModel):
    id: str
    question: str
    category: Category
    expected: Expected
    relevant_sections: list[str] = []   # citation ids needed for a complete answer
    reference_answer: str = ""          # what a good answer says
    key_facts: list[str] = []           # facts the answer must contain (checked by a judge)
    forbidden_claims: list[str] = []    # things the answer must NOT say
    needs_disclaimer: bool = False      # legal / medical topics
    notes: str = ""

    @model_validator(mode="after")
    def _consistent(self):
        if self.expected == "answer":
            if not (self.relevant_sections and self.reference_answer and self.key_facts):
                raise ValueError(f"{self.id}: answerable questions need sections, reference answer and key facts")
        elif self.relevant_sections:
            raise ValueError(f"{self.id}: only answerable questions have relevant sections")
        return self


def load_eval_set(path: Path = EVAL_FILE) -> list[EvalQuestion]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [EvalQuestion.model_validate(json.loads(ln)) for ln in lines]
