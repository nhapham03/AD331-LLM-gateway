from typing import TypedDict


class TokenPricing(TypedDict):
    input_usd_per_million: float
    output_usd_per_million: float


# Prices are USD per 1,000,000 tokens. Confirm these rates and model IDs
# against your provider accounts before relying on them for billing.
MODEL_PRICING: dict[str, TokenPricing] = {
    "gpt-4o-mini": {
        "input_usd_per_million": 0.15,
        "output_usd_per_million": 0.60,
    },
    "gpt-4o": {
        "input_usd_per_million": 2.50,
        "output_usd_per_million": 10.00,
    },
    "claude-haiku-5-5": {
        "input_usd_per_million": 0.10,
        "output_usd_per_million": 0.50,
    },
    "claude-sonnet-5-5": {
        "input_usd_per_million": 2.00,
        "output_usd_per_million": 10.00,
    },
}

OPENAI_MODELS = frozenset({"gpt-4o-mini", "gpt-4o"})
ANTHROPIC_MODELS = frozenset({"claude-haiku-5-5", "claude-sonnet-5-5"})