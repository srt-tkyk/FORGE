"""Abstract base class for all models."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @abstractmethod
    def load(self, path: Path) -> None: ...
