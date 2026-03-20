"""Tests for Surrogate Model (Phase 2)."""

import numpy as np
import pytest

from forge.models.surrogate_model import SurrogateModel


class TestSurrogateModel:
    def test_fit_and_predict_shape(self):
        """Model outputs correct shapes for mu and sigma."""
        rng = np.random.default_rng(42)
        X_train = rng.random((15, 4))
        y_train = rng.random(15)

        model = SurrogateModel(training_iterations=20)
        model.fit(X_train, y_train)

        X_test = rng.random((5, 4))
        mu, sigma = model.predict_with_uncertainty(X_test)

        assert mu.shape == (5,)
        assert sigma.shape == (5,)

    def test_sigma_positive(self):
        """Sigma should always be non-negative."""
        rng = np.random.default_rng(42)
        X_train = rng.random((15, 4))
        y_train = rng.random(15)

        model = SurrogateModel(training_iterations=20)
        model.fit(X_train, y_train)

        X_test = rng.random((10, 4))
        _, sigma = model.predict_with_uncertainty(X_test)
        assert np.all(sigma >= 0)

    def test_uncertainty_higher_far_from_data(self):
        """Points far from training data should have higher uncertainty."""
        rng = np.random.default_rng(42)
        # Training data clustered near 0
        X_train = rng.random((20, 2)) * 0.1
        y_train = rng.random(20)

        model = SurrogateModel(training_iterations=30)
        model.fit(X_train, y_train)

        X_near = np.array([[0.05, 0.05]])
        X_far = np.array([[10.0, 10.0]])

        _, sigma_near = model.predict_with_uncertainty(X_near)
        _, sigma_far = model.predict_with_uncertainty(X_far)

        assert sigma_far[0] > sigma_near[0]

    def test_predict_returns_mean(self):
        """predict() should return the same as predict_with_uncertainty()[0]."""
        rng = np.random.default_rng(42)
        X_train = rng.random((15, 4))
        y_train = rng.random(15)

        model = SurrogateModel(training_iterations=20)
        model.fit(X_train, y_train)

        X_test = rng.random((5, 4))
        mu_direct = model.predict(X_test)
        mu_from_uncertainty, _ = model.predict_with_uncertainty(X_test)

        np.testing.assert_array_almost_equal(mu_direct, mu_from_uncertainty)

    def test_save_and_load(self, tmp_path):
        """Model can be saved and loaded with consistent predictions."""
        rng = np.random.default_rng(42)
        X_train = rng.random((15, 4))
        y_train = rng.random(15)

        model = SurrogateModel(training_iterations=20)
        model.fit(X_train, y_train)

        X_test = rng.random((5, 4))
        mu_before, sigma_before = model.predict_with_uncertainty(X_test)

        path = tmp_path / "surrogate.pt"
        model.save(path)

        model2 = SurrogateModel()
        model2.load(path)
        mu_after, sigma_after = model2.predict_with_uncertainty(X_test)

        np.testing.assert_array_almost_equal(mu_before, mu_after, decimal=4)
        np.testing.assert_array_almost_equal(sigma_before, sigma_after, decimal=4)

    def test_training_loss_decreases(self):
        """Training loss should generally decrease over iterations."""
        rng = np.random.default_rng(42)
        X_train = rng.random((20, 4))
        y_train = rng.random(20)

        model = SurrogateModel(training_iterations=50)
        losses = model.fit(X_train, y_train)

        assert len(losses) == 50
        # First loss should be higher than last (on average)
        assert losses[0] > losses[-1]
