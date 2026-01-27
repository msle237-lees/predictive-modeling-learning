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


@dataclass
class RegressionMetrics:
    """Container for common regression evaluation metrics."""
    r2: float
    mae: float
    mse: float
    rmse: float

class PolynomialModel:
    """
    Polynomial regression wrapper.

    Provides:
    - train/test split + fit
    - metrics evaluation
    - export helpers (coeffs, intercept, equation, predictions, metrics)
    - basic diagnostic plots

    Notes:
    - Input dataframe should already be cleaned and numeric-ready.
    - If categorical columns exist, encode them before training (e.g., one-hot).
    """

    def __init__(self, degree: int = 2):
        """Initialize the polynomial regression model wrapper."""
        self.degree = degree
        self.poly_features = PolynomialFeatures(degree=self.degree)
        self.model = LinearRegression()
        self.is_fitted = False

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
        ):
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
        Train the polynomial regression model.

        @param data Input dataset as a pandas DataFrame.
        @param target_column Name of the target column (dependent variable).
        @param test_size Fraction of data used for test split.
        @param random_state Random seed for reproducibility.
        """
        self.target_column = target_column

        self.X_train, self.X_test, self.y_train, self.y_test = self._load_data(
            data, target_column, test_size, random_state
        )

        # Transform features to polynomial features
        X_train_poly = self.poly_features.fit_transform(self.X_train)
        X_test_poly = self.poly_features.transform(self.X_test)

        self.feature_names = self.poly_features.get_feature_names_out(self.X_train.columns)

        # Fit the linear regression model on polynomial features
        self.model.fit(X_train_poly, self.y_train)

        # Predict on test set
        self.y_pred = self.model.predict(X_test_poly)

        self.is_fitted = True

    def _require_trained(self):
        """Ensure the model has been trained before accessing results."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before accessing this property.")
        
    def evaluate(self) -> RegressionMetrics:
        """
        Evaluate the trained model on the test set.

        @return RegressionMetrics dataclass with evaluation metrics.
        """
        self._require_trained()

        r2 = self.model.score(self.poly_features.transform(self.X_test), self.y_test)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)

        return RegressionMetrics(r2=r2, mae=mae, mse=mse, rmse=rmse)
    
    def export_coefficients(self) -> Dict[str, float]:
        """
        Export the model coefficients.

        @return Dictionary mapping feature names to their coefficients.
        """
        self._require_trained()

        coeffs = self.model.coef_
        return {name: coeff for name, coeff in zip(self.feature_names, coeffs)}
    
    def export_intercept(self) -> float:
        """
        Export the model intercept.

        @return Intercept value.
        """
        self._require_trained()
        return float(self.model.intercept_)
    
    def export_coefficient_table(self) -> pd.DataFrame:
        """
        Export coefficients as a pandas DataFrame.

        @return DataFrame with columns: feature, coefficient.
        """
        self._require_trained()

        coeffs = self.export_coefficients()
        return pd.DataFrame(list(coeffs.items()), columns=["feature", "coefficient"])
    
    def export_equation(self) -> str:
        """
        Export the regression equation as a human-readable string.

        @return Regression equation string.
        """
        self._require_trained()

        terms = []
        for feature, coeff in self.export_coefficients().items():
            terms.append(f"({coeff:.4f} * {feature})")
        
        equation = " + ".join(terms)
        equation = f"y = {equation} + ({self.export_intercept():.4f})"
        return equation
    
    def export_predictions(self) -> pd.DataFrame:
        """
        Export predictions along with actual values and residuals.

        @return DataFrame with columns: actual, predicted, residual.
        """
        self._require_trained()

        residuals = self.y_test - self.y_pred
        return pd.DataFrame({
            "actual": self.y_test,
            "predicted": self.y_pred,
            "residual": residuals
        })
    
    def plot_predicted_vs_actual(self):
        """
        Plot predicted vs actual values.
        """
        self._require_trained()

        plt.figure(figsize=(8, 6))
        plt.scatter(self.y_test, self.y_pred, alpha=0.7)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--')
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Predicted vs Actual Values")
        plt.grid()
        plt.show()

    def plot_residuals_vs_predicted(self):
        """
        Plot residuals vs predicted values.
        """
        self._require_trained()

        residuals = self.y_test - self.y_pred

        plt.figure(figsize=(8, 6))
        plt.scatter(self.y_pred, residuals, alpha=0.7)
        plt.axhline(0, color='r', linestyle='--')
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Predicted Values")
        plt.grid()
        plt.show()

    def plot_residual_distribution(self, bins: int = 30):
        """
        Plot histogram of residuals.

        @param bins Number of bins for the histogram.
        """
        self._require_trained()

        residuals = self.y_test - self.y_pred

        plt.figure(figsize=(8, 6))
        plt.hist(residuals, bins=bins, edgecolor='black', alpha=0.7)
        plt.xlabel("Residual")
        plt.ylabel("Frequency")
        plt.title("Residual Distribution")
        plt.grid()
        plt.show()

if __name__ == "__main__":
    # Example usage

    # Create a sample dataset
    np.random.seed(42)
    X = pd.DataFrame({
        "feature1": np.random.rand(100),
        "feature2": np.random.rand(100),
    })
    y = 3 + 2 * X["feature1"]**2 + 4 * X["feature2"] + np.random.randn(100) * 0.1
    data = X.copy()
    data["target"] = y

    # Initialize and train the model
    model = PolynomialModel(degree=2)
    model.train(data, target_column="target")

    # Evaluate the model
    metrics = model.evaluate()
    print("Evaluation Metrics:")
    print(metrics)

    # Export coefficients and equation
    coeffs = model.export_coefficients()
    equation = model.export_equation()
    print("\nCoefficients:")
    print(coeffs)
    print("\nRegression Equation:")
    print(equation)

    # Export predictions
    predictions_df = model.export_predictions()
    print("\nPredictions:")
    print(predictions_df.head())

    # Plot diagnostics
    model.plot_predicted_vs_actual()
    model.plot_residuals_vs_predicted()
    model.plot_residual_distribution()
    
