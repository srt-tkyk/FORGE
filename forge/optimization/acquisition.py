"""Acquisition functions: UCB and EI."""

import numpy as np
from scipy.stats import norm


def ucb(mu: np.ndarray, sigma: np.ndarray, kappa: float = 2.0) -> np.ndarray:
    """Upper Confidence Bound: α = μ + κσ"""
    return mu + kappa * sigma


def ei(mu: np.ndarray, sigma: np.ndarray, y_best: float) -> np.ndarray:
    """Expected Improvement."""
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (mu - y_best) / sigma
        result = (mu - y_best) * norm.cdf(z) + sigma * norm.pdf(z)
        # Where sigma == 0, EI is 0
        result = np.where(sigma > 1e-10, result, 0.0)
    return result
