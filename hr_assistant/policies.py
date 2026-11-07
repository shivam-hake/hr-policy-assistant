"""Policy documents -> LangChain Documents, one per numbered rule.

Each policy is a Markdown file with a small front-matter block and numbered
headings:

    ## 3. Earned leave            <- topic
    ### 3.2 Carry forward         <- section (one rule)

LangChain's MarkdownHeaderTextSplitter splits on these headings and records
them in each Document's metadata. We then add a stable citation id such as
"LEAVE-3.2". Splitting on headings (instead of every N characters) means each
chunk holds one complete rule, and every answer can cite where it came from.

The output is plain list[Document], the standard LangChain type, so it plugs
straight into any embedding model, vector store or retriever.
"""
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

REQUIRED_META = {"policy_id", "title", "version", "effective"}
NUMBERED = re.compile(r"^(\d+(?:\.\d+)?)\.?\s+(.+)$")  # "3. Earned leave" / "3.2 Carry forward"

_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "doc_title"), ("##", "topic"), ("###", "section")],
    strip_headers=True,
)


class PolicyFormatError(ValueError):
    pass


def _front_matter(raw: str, name: str) -> tuple[dict[str, str], str]:
    raw = raw.lstrip("\ufeff")
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw, re.DOTALL)
    if not m:
        raise PolicyFormatError(f"{name}: missing front matter (--- block at the top)")
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    missing = REQUIRED_META - meta.keys()
    if missing:
        raise PolicyFormatError(f"{name}: front matter missing {sorted(missing)}")
    return meta, raw[m.end():]


def _numbered(header: str, name: str) -> tuple[str, str]:
    m = NUMBERED.match(header.strip())
    if not m:
        raise PolicyFormatError(f"{name}: heading {header!r} must start with a number like '3.' or '3.2'")
    return m.group(1), m.group(2)


def parse_policy(raw: str, name: str) -> list[Document]:
    meta, body = _front_matter(raw, name)
    docs = []
    for chunk in _splitter.split_text(body):
        text = " ".join(chunk.page_content.split())
        if not text or "topic" not in chunk.metadata:
            continue
        topic_no, topic = _numbered(chunk.metadata["topic"], name)
        if "section" in chunk.metadata:
            number, title = _numbered(chunk.metadata["section"], name)
            if not number.startswith(f"{topic_no}."):
                raise PolicyFormatError(f"{name}: section {number} is not under topic {topic_no}")
        else:  # text directly under a topic heading
            number, title = topic_no, topic
        docs.append(Document(
            page_content=text,
            metadata={
                "citation": f"{meta['policy_id']}-{number}",
                "policy_id": meta["policy_id"],
                "policy_title": meta["title"],
                "version": meta["version"],
                "effective": meta["effective"],
                "section": number,
                "topic": topic,
                "title": title,
                "heading": f"{meta['title']} > {number} {title}",
                "source": name,
            },
        ))
    if not docs:
        raise PolicyFormatError(f"{name}: no numbered sections found")
    return docs


def load_policies(policies_dir: str | Path) -> list[Document]:
    files = sorted(Path(policies_dir).glob("*.md"))
    if not files:
        raise PolicyFormatError(f"No .md policy files in {policies_dir}")
    docs = [d for f in files for d in parse_policy(f.read_text(encoding="utf-8"), f.name)]
    seen: set[str] = set()
    for d in docs:
        if d.metadata["citation"] in seen:
            raise PolicyFormatError(f"Duplicate citation id {d.metadata['citation']}")
        seen.add(d.metadata["citation"])
    return docs
