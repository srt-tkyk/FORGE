"""Reward Model: Ordinal Regression (rank → continuous latent score)."""

import pickle
import warnings
from pathlib import Path

import numpy as np

from forge.models.base import BaseModel

try:
    import mord

    HAS_MORD = True
except ImportError:
    HAS_MORD = False


class RewardModel(BaseModel):
    """Ordinal regression model mapping rank labels to continuous latent scores."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit ordinal regression. y should be integer-encoded ranks."""
        if len(X) < 10:
            warnings.warn(
                f"データが少ない ({len(X)} 件)。結果が不安定になる可能性があります。",
                stacklevel=2,
            )

        if HAS_MORD:
            self._model = mord.LogisticAT(alpha=self.alpha)
        else:
            warnings.warn(
                "mord が利用できません。sklearn LogisticRegression で代替します。",
                stacklevel=2,
            )
            self._model = _FallbackOrdinalModel(alpha=self.alpha)

        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous latent scores (y_hat)."""
        if self._model is None:
            raise RuntimeError("モデルが未学習です。先に fit() を実行してください")

        if HAS_MORD:
            # mord returns the latent continuous value via decision_function
            # For LogisticAT, predict gives ordinal class; we want the continuous score
            return self._model.predict(X).astype(float)
        else:
            return self._model.predict(X)

    def predict_latent(self, X: np.ndarray) -> np.ndarray:
        """Predict raw latent scores (decision function values)."""
        if self._model is None:
            raise RuntimeError("モデルが未学習です。先に fit() を実行してください")

        if HAS_MORD and hasattr(self._model, "decision_function"):
            return self._model.decision_function(X)
        else:
            return self.predict(X)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "alpha": self.alpha}, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._model = data["model"]
        self.alpha = data["alpha"]


class _FallbackOrdinalModel:
    """Fallback using sklearn LogisticRegression when mord is unavailable."""

    def __init__(self, alpha: float = 1.0):
        from sklearn.linear_model import LogisticRegression

        self.clf = LogisticRegression(C=1.0 / alpha, max_iter=1000, multi_class="multinomial")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.clf.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return expected value as continuous score."""
        proba = self.clf.predict_proba(X)
        classes = self.clf.classes_.astype(float)
        return proba @ classes
