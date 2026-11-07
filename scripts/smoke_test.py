"""Smoke test: one real call to each configured model.

Run from the project root:  uv run python -m scripts.smoke_test
It prints the reply, token usage and latency: the same three numbers
our telemetry will track for every agent step later.
"""
import time

from hr_assistant.llm import describe, get_llm


def main() -> None:
    print("Config:", describe())
    for role in ("main", "fast"):
        llm = get_llm(role)
        start = time.perf_counter()
        reply = llm.invoke("Reply with exactly one word: OK")
        latency = time.perf_counter() - start
        usage = reply.usage_metadata or {}
        print(
            f"[{role}] reply={reply.text.strip()!r} "
            f"input_tokens={usage.get('input_tokens')} "
            f"output_tokens={usage.get('output_tokens')} "
            f"latency={latency:.2f}s"
        )


if __name__ == "__main__":
    main()
