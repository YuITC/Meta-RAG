import numpy as np
from dataclasses import dataclass, field

# Three retrieval configurations
CONFIGS: dict[str, dict] = {
    "A": {"top_k": 5, "chunk_size": 256, "rerank": False},
    "B": {"top_k": 10, "chunk_size": 512, "rerank": False},
    "C": {"top_k": 10, "chunk_size": 512, "rerank": True},
}

CONFIG_NAMES = list(CONFIGS.keys())


@dataclass
class ThompsonSamplingBandit:
    """Thompson Sampling bandit with Beta(1,1) priors over retrieval configs."""

    alpha: dict[str, float] = field(default_factory=lambda: {c: 1.0 for c in CONFIG_NAMES})
    beta: dict[str, float] = field(default_factory=lambda: {c: 1.0 for c in CONFIG_NAMES})

    def select_config(self, exclude: str | None = None) -> str:
        candidates = [c for c in CONFIG_NAMES if c != exclude]
        samples = {
            c: float(np.random.beta(self.alpha[c], self.beta[c])) for c in candidates
        }
        return max(samples, key=samples.__getitem__)

    def update(self, config: str, utility: float, threshold: float = 0.7) -> None:
        if utility >= threshold:
            self.alpha[config] += 1.0
        else:
            self.beta[config] += 1.0

    def stats(self) -> dict[str, dict]:
        result = {}
        for c in CONFIG_NAMES:
            a, b = self.alpha[c], self.beta[c]
            result[c] = {
                "alpha": a,
                "beta": b,
                "win_rate": round(a / (a + b), 4),
                "trials": int(a + b - 2),  # subtract Beta(1,1) prior
            }
        return result


def compute_utility(
    faithfulness: float,
    cost: float,
    latency: float,
    lambda1: float = 0.3,
    lambda2: float = 0.2,
) -> float:
    """
    Utility = Faithfulness - λ₁ × Cost_norm - λ₂ × Latency_norm
    Cost_norm  = cost / 0.005
    Latency_norm = latency / 15.0
    """
    cost_norm = cost / 0.005
    latency_norm = latency / 15.0
    utility = faithfulness - lambda1 * cost_norm - lambda2 * latency_norm
    return round(max(0.0, min(1.0, utility)), 4)


def estimate_cost(input_chars: int, output_chars: int) -> float:
    """
    Rough cost estimate using Gemini 2.5 Flash pricing.
    ~4 chars per token. Input: $0.075/1M, Output: $0.30/1M tokens.
    """
    input_tokens = input_chars / 4
    output_tokens = output_chars / 4
    return input_tokens * 0.075 / 1_000_000 + output_tokens * 0.30 / 1_000_000
