"""LLM prices in USD per 1 million tokens: (input, output).

Prices change, so treat this table as configuration and re-check it when you
change models. On a free tier you pay nothing; the cost column then shows
what the same traffic would cost on a paid plan, which is what a company
would care about. Hidden reasoning tokens are billed as output tokens.

Groq rates checked Nov 2026 (console.groq.com/docs/models).
Gemini rates: VERIFY at ai.google.dev/gemini-api/docs/pricing before quoting them.
"""
PRICES_PER_MILLION: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
}


def cost_usd(model: str | None, input_tokens: int, output_tokens: int) -> float | None:
    """Cost of one call, or None if the model has no known price."""
    if not model:
        return None
    price = PRICES_PER_MILLION.get(model) or PRICES_PER_MILLION.get(model.split("/")[-1])
    if price is None:
        # Some providers return names like "models/gemini-2.5-flash"
        price = next((p for name, p in PRICES_PER_MILLION.items() if model.endswith(name)), None)
    if price is None:
        return None
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000
