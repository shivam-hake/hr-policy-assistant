"""Try retrieval by hand: shows the top sections for a question with similarity scores.
Run:  uv run python -m scripts.search "Can I claim my home internet bill?"
"""
import sys

from hr_assistant.config import get_settings
from hr_assistant.vectorstore import get_vectorstore


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m scripts.search "your question"')
        return
    question = " ".join(sys.argv[1:])
    store = get_vectorstore()
    results = store.similarity_search_with_relevance_scores(question, k=get_settings().retrieval_k)
    print(f"\nQ: {question}\n")
    for doc, score in results:
        text = doc.page_content.split("\n", 1)[1]
        print(f"{score:.3f}  {doc.metadata['citation']:<11} {doc.metadata['title']}")
        print(f"       {text[:110]}{'...' if len(text) > 110 else ''}\n")


if __name__ == "__main__":
    main()
