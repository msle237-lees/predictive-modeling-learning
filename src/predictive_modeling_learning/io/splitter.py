# io/splitter.py
from dataclasses import dataclass

import numpy as np


@dataclass
class DataBundle:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray | None
    y_test: np.ndarray | None
    feature_names: list[str]
