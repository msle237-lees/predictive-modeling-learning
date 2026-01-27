"""
linear_model.py
Linear regression model wrapper with training, evaluation, exports, and diagnostics.

Exports:
- Coefficients (per feature)
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
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class RegressionMetrics:
    """Container for common regression evaluation metrics."""
    r2: float
    mae: float
    mse: float
    rmse: float


class LinearModel:
    """
    Simple linear regression wrapper.

    Provides:
    - train/test split + fit
    - metrics evaluation
    - export helpers (coeffs, intercept, equation, predictions, metrics)
    - basic diagnostic plots

    Notes:
    - Input dataframe should already be cleaned and numeric-ready.
    - If categorical columns exist, encode them before training (e.g., one-hot).
    """

    def __init__(self):
        """Initialize the linear regression model wrapper."""
        self.model = LinearRegression()
        self.is_trained = False

        self.target_column: Optional[str] = None
        self.feature_names: Optional[np.ndarray] = None

        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.y_pred: Optional[np.ndarray] = None

    def _load_data(
        self,
        data: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split dataframe into train/test.

        @param data Input dataset as a pandas DataFrame.
        @param target_column Name of the target column (dependent variable).
        @param test_size Fraction of data used for test split.
        @param random_state Random seed for reproducibility.
        @return Tuple of X_train, X_test, y_train, y_test.
        """
        if target_column not in data.columns:
            raise ValueError(f"target_column '{target_column}' not found in dataframe columns.")

        X = data.drop(columns=[target_column])
        y = data[target_column]

        if X.shape[1] == 0:
            raise ValueError("No feature columns found after dropping target column.")

        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def train(
        self,
        data: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Train the linear regression model.

        @param data Input dataset as a pandas DataFrame.
        @param target_column Name of the target column (dependent variable).
        @param test_size Fraction of data used for test split.
        @param random_state Random seed for reproducibility.
        """
        X_train, X_test, y_train, y_test = self._load_data(
            data=data,
            target_column=target_column,
            test_size=test_size,
            random_state=random_state,
        )

        self.model.fit(X_train, y_train)

        self.is_trained = True
        self.target_column = target_column
        self.feature_names = getattr(self.model, "feature_names_in_", X_train.columns.to_numpy())

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # Cache predictions for exports and plots
        self.y_pred = self.model.predict(self.X_test)

    def _require_trained(self):
        """
        Ensure model is trained before performing operations.

        @raises RuntimeError if model not trained.
        """
        if not self.is_trained or self.X_test is None or self.y_test is None or self.y_pred is None:
            raise RuntimeError("Model must be trained before evaluation/export/plotting.")

    def evaluate(self) -> RegressionMetrics:
        """
        Evaluate model on the test set and print metrics.

        @return RegressionMetrics containing r2, mae, mse, rmse.
        """
        self._require_trained()

        r2 = float(self.model.score(self.X_test, self.y_test))
        mae = float(mean_absolute_error(self.y_test, self.y_pred))
        mse = float(mean_squared_error(self.y_test, self.y_pred))
        rmse = float(np.sqrt(mse))

        metrics = RegressionMetrics(r2=r2, mae=mae, mse=mse, rmse=rmse)

        print(f"Model metrics:")
        print(f"  R^2 : {metrics.r2:.4f}")
        print(f"  MAE : {metrics.mae:.4f}")
        print(f"  MSE : {metrics.mse:.4f}")
        print(f"  RMSE: {metrics.rmse:.4f}")

        return metrics

    def export_metrics(self) -> Dict[str, float]:
        """
        Export evaluation metrics as a dictionary.

        @return Dict with keys: r2, mae, mse, rmse.
        """
        metrics = self.evaluate()
        return {
            "r2": metrics.r2,
            "mae": metrics.mae,
            "mse": metrics.mse,
            "rmse": metrics.rmse,
        }

    def export_intercept(self) -> float:
        """
        Export the intercept term of the trained model.

        @return Intercept as float.
        """
        self._require_trained()
        return float(self.model.intercept_)

    def export_coefficients(self) -> pd.DataFrame:
        """
        Export coefficients as a dataframe.

        @return DataFrame with columns: Feature, Coefficient.
        """
        self._require_trained()

        # feature_names is stored from training time
        feature_names = (
            self.feature_names
            if self.feature_names is not None
            else np.array([f"feature_{i}" for i in range(len(self.model.coef_))])
        )

        coeffs = pd.DataFrame(
            {
                "Feature": feature_names,
                "Coefficient": self.model.coef_,
            }
        )

        # Optional: sort by absolute magnitude (useful for quick inspection)
        coeffs["AbsCoefficient"] = coeffs["Coefficient"].abs()
        coeffs = coeffs.sort_values("AbsCoefficient", ascending=False).drop(columns=["AbsCoefficient"]).reset_index(drop=True)

        return coeffs

    def export_regression_equation(self, precision: int = 6) -> str:
        """
        Export a human-readable regression equation.

        Example:
            y = 1.23 + 0.45*feature_a - 2.10*feature_b

        @param precision Number of decimal places for coefficients.
        @return Equation string.
        """
        self._require_trained()

        intercept = self.export_intercept()
        coeffs = self.model.coef_

        feature_names = (
            self.feature_names
            if self.feature_names is not None
            else np.array([f"feature_{i}" for i in range(len(coeffs))])
        )

        # Build equation terms
        terms = [f"{intercept:.{precision}f}"]
        for name, coef in zip(feature_names, coeffs):
            sign = "+" if coef >= 0 else "-"
            terms.append(f" {sign} {abs(coef):.{precision}f}*{name}")

        y_name = self.target_column if self.target_column else "y"
        equation = f"{y_name} = " + "".join(terms)
        return equation

    def export_predictions(self) -> pd.DataFrame:
        """
        Export actual vs predicted values and residuals.

        @return DataFrame with columns: actual, predicted, residual.
        """
        self._require_trained()

        df = pd.DataFrame(
            {
                "actual": self.y_test.to_numpy(),
                "predicted": self.y_pred,
            }
        )
        df["residual"] = df["actual"] - df["predicted"]
        return df

    def plot_predicted_vs_actual(self):
        """
        Plot predicted vs actual values.

        Shows how close predictions are to true values.
        """
        self._require_trained()

        plt.figure()
        plt.scatter(self.y_test, self.y_pred)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Predicted vs Actual")
        plt.grid(True)
        plt.show()

    def plot_residuals_vs_predicted(self):
        """
        Plot residuals vs predicted values.

        Useful to check:
        - non-linearity
        - heteroscedasticity (error variance changing with prediction size)
        """
        self._require_trained()

        residuals = self.y_test.to_numpy() - self.y_pred

        plt.figure()
        plt.scatter(self.y_pred, residuals)
        plt.axhline(0)
        plt.xlabel("Predicted")
        plt.ylabel("Residual (Actual - Predicted)")
        plt.title("Residuals vs Predicted")
        plt.grid(True)
        plt.show()

    def plot_residual_distribution(self, bins: int = 30):
        """
        Plot histogram of residuals.

        Useful to see if residuals look roughly normal.
        """
        self._require_trained()

        residuals = self.y_test.to_numpy() - self.y_pred

        plt.figure()
        plt.hist(residuals, bins=bins)
        plt.xlabel("Residual")
        plt.ylabel("Frequency")
        plt.title("Residual Distribution")
        plt.grid(True)
        plt.show()
