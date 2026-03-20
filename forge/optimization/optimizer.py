"""Optimization: find best action using acquisition function."""

from __future__ import annotations

import numpy as np
from scipy.optimize import differential_evolution

from forge.optimization.acquisition import ucb, ei


def optimize_acquisition(
    surrogate_model,
    c_vec: np.ndarray,
    bounds: list[tuple[float, float]],
    acq_func: str = "ucb",
    kappa: float = 2.0,
    y_best: float = 0.0,
    method_kwargs: dict | None = None,
) -> dict:
    """Find the action that maximizes the acquisition function.

    Args:
        surrogate_model: Trained SurrogateModel with predict_with_uncertainty
        c_vec: Condition vector (1D)
        bounds: Action bounds [(min, max), ...]
        acq_func: "ucb" or "ei"
        kappa: UCB exploration parameter
        y_best: Best observed y_hat (for EI)
        method_kwargs: Extra kwargs for differential_evolution

    Returns:
        dict with keys: a_vec, mu, sigma, alpha, result
    """
    n_c = len(c_vec)

    def objective(a_vec):
        # Build full feature vector [C, A]
        x = np.concatenate([c_vec, a_vec]).reshape(1, -1)
        mu_val, sigma_val = surrogate_model.predict_with_uncertainty(x)
        if acq_func == "ucb":
            alpha_val = ucb(mu_val, sigma_val, kappa=kappa)
        elif acq_func == "ei":
            alpha_val = ei(mu_val, sigma_val, y_best=y_best)
        else:
            raise ValueError(f"Unknown acquisition function: {acq_func}")
        # Minimize negative acquisition (maximize acquisition)
        return -alpha_val[0]

    kwargs = {
        "maxiter": 1000,
        "popsize": 15,
        "seed": 42,
    }
    if method_kwargs:
        kwargs.update(method_kwargs)

    result = differential_evolution(objective, bounds=bounds, **kwargs)

    # Get final predictions at optimum
    a_opt = result.x
    x_opt = np.concatenate([c_vec, a_opt]).reshape(1, -1)
    mu_opt, sigma_opt = surrogate_model.predict_with_uncertainty(x_opt)

    if acq_func == "ucb":
        alpha_opt = ucb(mu_opt, sigma_opt, kappa=kappa)
    else:
        alpha_opt = ei(mu_opt, sigma_opt, y_best=y_best)

    return {
        "a_vec": a_opt,
        "mu": mu_opt[0],
        "sigma": sigma_opt[0],
        "alpha": alpha_opt[0],
        "result": result,
    }
