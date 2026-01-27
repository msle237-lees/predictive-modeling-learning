"""
./src/predictive_modeling_learning/models/regression/polynomial.py
Polynomial regression model wrapper with training, evaluation, exports, and diagnostics.

Exports:
- Coefficients (per feature and degree)
- Intercept
- Regression equation (human-readable)
- Metrics (R2, MAE, MSE, RMSE)
- Predictions (actual, predicted, residual)

Plots:
- Predicted vs Actual
- Residuals vs Predicted
- Residual distribution (histogram)

Author: Michael Lees
Date: 2026-01-26
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error