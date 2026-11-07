"""Cost and latency of questions asked through the app (API/UI).
Run:  uv run python -m scripts.telemetry_report"""
from hr_assistant.config import get_settings
from hr_assistant.telemetry import TelemetryStore


def main() -> None:
    t = TelemetryStore(get_settings().telemetry_db_path).summary()
    if not t["requests"]:
        print("No questions recorded yet. Ask some through the API or UI.")
        return
    cost = t["avg_cost_per_request_usd"]
    print(f"Questions: {t['requests']}  {t['by_status']}")
    print(f"Latency: avg {t['avg_latency_ms']} ms, p95 {t['p95_latency_ms']} ms")
    print(f"LLM calls: {t['llm_calls']}, avg cost per question: {'n/a' if cost is None else f'${cost:.6f}'}\n")
    for s in t["steps"]:
        print(f"{s['step']:<12} calls {s['calls']:<4} errors {s['errors']:<3} avg {s['avg_latency_ms']} ms  "
              f"p95 {s['p95_latency_ms']} ms  tokens {s['avg_input_tokens']} in / {s['avg_output_tokens']} out")


if __name__ == "__main__":
    main()
