"""Surrogate Model: Gaussian Process (GPyTorch ExactGP)."""

from pathlib import Path

import numpy as np
import torch
import gpytorch

from forge.models.base import BaseModel


class ExactGPModel(gpytorch.models.ExactGP):
    """Standard ExactGP with Matern kernel."""

    def __init__(self, train_x, train_y, likelihood, nu=2.5):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=nu)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SurrogateModel(BaseModel):
    """GP-based surrogate model f(C,A) → (μ, σ)."""

    def __init__(self, nu: float = 2.5, noise_variance: float = 0.01,
                 training_iterations: int = 100):
        self.nu = nu
        self.noise_variance = noise_variance
        self.training_iterations = training_iterations
        self._model = None
        self._likelihood = None
        self._train_x = None
        self._train_y = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> list[float]:
        """Fit GP model. Returns list of training losses."""
        train_x = torch.tensor(X, dtype=torch.float64)
        train_y = torch.tensor(y, dtype=torch.float64)

        self._train_x = train_x
        self._train_y = train_y

        self._likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self._likelihood.noise = self.noise_variance

        self._model = ExactGPModel(train_x, train_y, self._likelihood, nu=self.nu)

        # Use float64 for both
        self._model.double()
        self._likelihood.double()

        # Training mode
        self._model.train()
        self._likelihood.train()

        optimizer = torch.optim.Adam(self._model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._likelihood, self._model)

        losses = []
        for i in range(self.training_iterations):
            optimizer.zero_grad()
            output = self._model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict mean values."""
        mu, _ = self.predict_with_uncertainty(X)
        return mu

    def predict_with_uncertainty(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict (mu, sigma)."""
        if self._model is None:
            raise RuntimeError("モデルが未学習です。先に fit() を実行してください")

        self._model.eval()
        self._likelihood.eval()

        test_x = torch.tensor(X, dtype=torch.float64)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self._likelihood(self._model(test_x))
            mu = pred.mean.numpy()
            sigma = pred.stddev.numpy()

        return mu, sigma

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self._model.state_dict(),
            "likelihood_state": self._likelihood.state_dict(),
            "train_x": self._train_x,
            "train_y": self._train_y,
            "nu": self.nu,
            "noise_variance": self.noise_variance,
            "training_iterations": self.training_iterations,
        }, path)

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, weights_only=False)
        self.nu = checkpoint["nu"]
        self.noise_variance = checkpoint["noise_variance"]
        self.training_iterations = checkpoint["training_iterations"]
        self._train_x = checkpoint["train_x"]
        self._train_y = checkpoint["train_y"]

        self._likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self._model = ExactGPModel(
            self._train_x, self._train_y, self._likelihood, nu=self.nu
        )
        self._model.double()
        self._likelihood.double()

        self._model.load_state_dict(checkpoint["model_state"])
        self._likelihood.load_state_dict(checkpoint["likelihood_state"])
