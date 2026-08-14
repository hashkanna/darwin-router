"""Hardcoded $/1M-token estimates. Approximate on purpose — labelled as
estimates in the UI. Counterfactual = what the request would have cost on
qwen-max."""

# model -> ($/1M input tokens, $/1M output tokens)
PRICES = {
    "qwen-flash": (0.05, 0.40),
    "qwen-plus": (0.40, 1.20),
    "qwen-max": (1.60, 6.40),
    "qwen-vl-max": (0.80, 3.20),
    # SIE lanes (guide: 27B is ~10x the 4B; absolute numbers are rough)
    "Qwen/Qwen3.5-4B": (0.04, 0.16),
    "Qwen/Qwen3.6-27B": (0.40, 1.60),
}

COUNTERFACTUAL_MODEL = "qwen-max"

# $ per generation for visual models — rough estimates
FLAT = {
    "z-image-turbo": 0.01,
    "qwen-image-3.0-pro": 0.06,
    "wan2.7-image": 0.04,
    "happyhorse-1.1-t2v": 0.20,
}


def flat_cost(model: str) -> float:
    return FLAT.get(model, 0.05)


def estimate(model: str, tokens_in: int, tokens_out: int) -> float:
    p_in, p_out = PRICES.get(model, PRICES[COUNTERFACTUAL_MODEL])
    return (tokens_in * p_in + tokens_out * p_out) / 1_000_000


def counterfactual(tokens_in: int, tokens_out: int) -> float:
    return estimate(COUNTERFACTUAL_MODEL, tokens_in, tokens_out)
