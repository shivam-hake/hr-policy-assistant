"""Build (or rebuild) the policy index.
Run:  uv run python -m scripts.ingest
The first run downloads the embedding model (~130 MB) into data/runtime/models."""
import time

from hr_assistant.config import get_settings
from hr_assistant.vectorstore import ingest


def main() -> None:
    s = get_settings()
    print(f"Embedding model: {s.embedding_model}")
    start = time.perf_counter()
    store = ingest()
    print(f"Indexed {store._collection.count()} sections into {s.chroma_dir} "
          f"in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
