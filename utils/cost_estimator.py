COST_PER_1K_TOKENS = {
    "OpenAI text-embedding-3-small": 0.00002,
    "OpenAI text-embedding-3-large": 0.00013,
    "OpenAI text-embedding-ada-002": 0.00010,
    "Cohere Embed English v3.0": 0.00010,
    "Cohere Embed Multilingual v3.0": 0.00010,
}

def estimate_tokens(texts: list[str]) -> int:
    return sum(len(t) for t in texts) // 4

def estimate_cost(texts: list[str], model_name: str) -> float:
    tokens = estimate_tokens(texts)
    rate = COST_PER_1K_TOKENS.get(model_name, 0.0)
    return round((tokens / 1000) * rate, 6)

def format_cost(cost: float) -> str:
    if cost == 0.0:
        return "Free (local)"
    if cost < 0.001:
        return f"${cost:.6f}"
    return f"${cost:.4f}"
