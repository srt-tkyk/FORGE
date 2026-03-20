"""Tests for acquisition functions and optimizer."""

import numpy as np
import pytest

from forge.optimization.acquisition import ucb, ei


class TestUCB:
    def test_basic_ucb(self):
        mu = np.array([1.0, 2.0, 3.0])
        sigma = np.array([0.5, 0.5, 0.5])
        result = ucb(mu, sigma, kappa=2.0)
        expected = np.array([2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_kappa_zero_returns_mu(self):
        mu = np.array([1.0, 2.0])
        sigma = np.array([10.0, 10.0])
        result = ucb(mu, sigma, kappa=0.0)
        np.testing.assert_array_almost_equal(result, mu)

    def test_higher_sigma_increases_ucb(self):
        mu = np.array([1.0, 1.0])
        sigma = np.array([0.1, 1.0])
        result = ucb(mu, sigma, kappa=2.0)
        assert result[1] > result[0]


class TestEI:
    def test_ei_positive_when_above_best(self):
        mu = np.array([5.0])
        sigma = np.array([1.0])
        y_best = 3.0
        result = ei(mu, sigma, y_best)
        assert result[0] > 0

    def test_ei_near_zero_when_below_best(self):
        mu = np.array([0.0])
        sigma = np.array([0.01])
        y_best = 10.0
        result = ei(mu, sigma, y_best)
        assert result[0] < 0.01

    def test_ei_zero_when_sigma_zero(self):
        mu = np.array([5.0])
        sigma = np.array([0.0])
        y_best = 3.0
        result = ei(mu, sigma, y_best)
        assert result[0] == 0.0

    def test_ei_shape(self):
        mu = np.array([1.0, 2.0, 3.0])
        sigma = np.array([0.5, 0.5, 0.5])
        result = ei(mu, sigma, y_best=1.5)
        assert result.shape == (3,)


class TestOptimizer:
    def test_optimize_acquisition_integration(self):
        """Integration test: optimizer finds reasonable action."""
        from forge.models.surrogate_model import SurrogateModel
        from forge.optimization.optimizer import optimize_acquisition

        rng = np.random.default_rng(42)

        # Train a simple surrogate
        X_train = rng.random((20, 4))
        y_train = X_train[:, 2] + X_train[:, 3]  # Score depends on actions
        model = SurrogateModel(training_iterations=30)
        model.fit(X_train, y_train)

        # Optimize with fixed conditions
        c_vec = np.array([0.5, 0.5])
        bounds = [(0.0, 1.0), (0.0, 1.0)]

        result = optimize_acquisition(
            surrogate_model=model,
            c_vec=c_vec,
            bounds=bounds,
            acq_func="ucb",
            kappa=2.0,
            method_kwargs={"maxiter": 50, "popsize": 10, "seed": 42},
        )

        assert "a_vec" in result
        assert "mu" in result
        assert "sigma" in result
        assert "alpha" in result
        assert len(result["a_vec"]) == 2
        # Actions should be within bounds
        assert 0.0 <= result["a_vec"][0] <= 1.0
        assert 0.0 <= result["a_vec"][1] <= 1.0
