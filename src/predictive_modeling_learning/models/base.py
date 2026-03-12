from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TrainResult:
    model_name: str
    hyperparams: dict
    metrics: dict
    artifact_path: str | None = None


class BaseModel(ABC):
    name: str  # e.g. "linear_regression"
    category: (
        str  # "regression" | "classification" | "clustering" | "time_series" | "neural"
    )

    @abstractmethod
    def build(self, **hyperparams) -> None:
        """Instantiate the underlying model with given hyperparams."""
        ...

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> TrainResult:
        """Fit the model, return a TrainResult."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions."""
        ...

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Return a dict of metric_name -> value."""
        ...

    def save(self, path: str) -> None:
        """Serialize model to disk (default: joblib)."""
        ...

    @classmethod
    def load(cls, path: str) -> "BaseModel":
        """Deserialize model from disk."""
        ...
