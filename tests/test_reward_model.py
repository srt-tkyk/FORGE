"""Tests for Reward Model (Phase 1)."""

import numpy as np
import pandas as pd
import pytest

from forge.models.reward_model import RewardModel
from forge.data.loader import load_dataset, get_feature_matrix
from forge.utils.config import load_schema, get_rank_mapping, REWARD_MODEL_PATH


class TestRewardModel:
    def test_fit_and_predict_shape(self):
        """Model outputs correct shape."""
        rng = np.random.default_rng(42)
        X = rng.random((20, 4))
        y = rng.choice([0, 1, 2], size=20)

        model = RewardModel(alpha=1.0)
        model.fit(X, y)

        preds = model.predict(X)
        assert preds.shape == (20,)

    def test_predict_values_ordered(self):
        """Higher rank class should tend to get higher predictions."""
        rng = np.random.default_rng(42)
        n = 60
        X = rng.random((n, 4))
        # Make y correlated with X[:,0]
        score = X[:, 0] * 3 + rng.normal(0, 0.3, n)
        y = np.digitize(score, bins=[1.0, 2.0])  # 0, 1, 2

        model = RewardModel(alpha=1.0)
        model.fit(X, y)
        preds = model.predict(X)

        # Mean prediction for class 2 should be > class 0
        mean_0 = preds[y == 0].mean()
        mean_2 = preds[y == 2].mean()
        assert mean_2 > mean_0

    def test_save_and_load(self, tmp_path):
        """Model can be saved and loaded."""
        rng = np.random.default_rng(42)
        X = rng.random((20, 4))
        y = rng.choice([0, 1, 2], size=20)

        model = RewardModel(alpha=1.0)
        model.fit(X, y)
        preds_before = model.predict(X)

        path = tmp_path / "reward_model.pkl"
        model.save(path)

        model2 = RewardModel()
        model2.load(path)
        preds_after = model2.predict(X)

        np.testing.assert_array_almost_equal(preds_before, preds_after)

    def test_few_samples_warning(self):
        """Warning issued when data < 10 samples."""
        X = np.random.default_rng(42).random((5, 4))
        y = np.array([0, 1, 2, 0, 1])

        model = RewardModel(alpha=1.0)
        with pytest.warns(UserWarning, match="データが少ない"):
            model.fit(X, y)

    def test_integration_with_dataset(self, tmp_project):
        """Full integration: load data, train, predict, save."""
        schema = load_schema()
        config_ranks = get_rank_mapping()
        df = load_dataset()

        ranked = df[df["h_rank"].notna() & (df["h_rank"] != "")]
        X = get_feature_matrix(ranked, schema)
        y = ranked["h_rank"].map(config_ranks).values.astype(int)

        model = RewardModel(alpha=1.0)
        model.fit(X, y)

        X_all = get_feature_matrix(df, schema)
        y_hat = model.predict(X_all)
        assert y_hat.shape == (len(df),)
        assert not np.any(np.isnan(y_hat))

        model.save(REWARD_MODEL_PATH)
        assert REWARD_MODEL_PATH.exists()
