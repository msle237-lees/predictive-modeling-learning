"""
./src/predictive_modeling_learning/cli/regression.py
Regression CLI commands.

Design:
- One parent group: regression
- One subgroup per model: linear (more later)
- Commands are "single-shot": load data -> train -> output/evaluate/plot

Author: Michael Lees
Date: 2026-01-26
"""

from __future__ import annotations

# Package imports
from predictive_modeling_learning.io import db
from predictive_modeling_learning.models.regression.linear import LinearModel

# Python imports
import json
from pathlib import Path
from typing import Optional

import click
import pandas as pd


# ----------------------------
# Helpers
# ----------------------------

def _load_dataframe(
    source: str,
    csv_path: Optional[str],
    table: Optional[str],
    schema: str,
    max_rows: int,
) -> pd.DataFrame:
    """
    @brief Load a dataframe from CSV or DB.

    @param source Data source ("csv" or "db").
    @param csv_path Path to CSV if source == "csv".
    @param table DB table name if source == "db".
    @param schema DB schema name.
    @param max_rows Max rows to fetch from DB (safety cap).
    @return pandas DataFrame.
    @raises click.ClickException on invalid args or load failure.
    """
    source = source.strip().lower()

    if source == "csv":
        if not csv_path:
            raise click.ClickException("Missing --csv for source=csv.")
        p = Path(csv_path).expanduser().resolve()
        if not p.exists():
            raise click.ClickException(f"CSV file not found: {p}")
        try:
            return pd.read_csv(p)
        except Exception as exc:
            raise click.ClickException(f"Failed to read CSV: {exc}") from exc

    if source == "db":
        if not table:
            raise click.ClickException("Missing --table for source=db.")
        try:
            return db.fetch_table_data(table_name=table, schema=schema, max_rows=max_rows)
        except Exception as exc:
            raise click.ClickException(f"Failed to fetch DB table '{schema}.{table}': {exc}") from exc

    raise click.ClickException("Invalid --source. Use 'csv' or 'db'.")


def _ensure_output_dir(out_dir: str) -> Path:
    """
    @brief Ensure output directory exists.

    @param out_dir Directory path.
    @return Path object.
    """
    p = Path(out_dir).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------
# Root group: regression
# ----------------------------

@click.group()
def regression():
    """Regression model CLI commands."""
    pass


@regression.command("list-tables")
def list_tables():
    """
    @brief List available tables from the configured DB API.

    Uses API_BASE_URL from your environment/.env. See io/db.py.
    """
    try:
        tables = db.get_tables()
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(json.dumps(tables, indent=2))


# ----------------------------
# Subgroup: linear regression
# ----------------------------

@regression.group()
def linear():
    """Linear regression model commands."""
    pass


@linear.command("train")
@click.option(
    "--source",
    type=click.Choice(["csv", "db"], case_sensitive=False),
    required=True,
    help="Where to load the dataset from.",
)
@click.option("--csv", "csv_path", type=click.Path(dir_okay=False), help="Path to a CSV file.")
@click.option("--table", type=str, help="DB table name (when --source db).")
@click.option("--schema", type=str, default="dbo", show_default=True, help="DB schema (when --source db).")
@click.option("--max-rows", type=int, default=100_000, show_default=True, help="Safety cap for DB fetch.")
@click.option("--target", "target_column", type=str, required=True, help="Target column name.")
@click.option("--test-size", type=float, default=0.2, show_default=True, help="Test split fraction.")
@click.option("--random-state", type=int, default=42, show_default=True, help="Random seed.")
def linear_train(
    source: str,
    csv_path: Optional[str],
    table: Optional[str],
    schema: str,
    max_rows: int,
    target_column: str,
    test_size: float,
    random_state: int,
):
    """
    @brief Train a linear regression model and print metrics.

    Note: This command trains and evaluates in one run (no persistence yet).
    """
    df = _load_dataframe(source, csv_path, table, schema, max_rows)

    if target_column not in df.columns:
        raise click.ClickException(f"Target column '{target_column}' not found in dataset.")

    model = LinearModel()
    model.train(data=df, target_column=target_column, test_size=test_size, random_state=random_state)

    # Prints metrics + returns RegressionMetrics (per your model wrapper)
    model.evaluate()


@linear.command("metrics")
@click.option("--source", type=click.Choice(["csv", "db"], case_sensitive=False), required=True)
@click.option("--csv", "csv_path", type=click.Path(dir_okay=False))
@click.option("--table", type=str)
@click.option("--schema", type=str, default="dbo", show_default=True)
@click.option("--max-rows", type=int, default=100_000, show_default=True)
@click.option("--target", "target_column", type=str, required=True)
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=False, help="Write metrics JSON to file.")
def linear_metrics(
    source: str,
    csv_path: Optional[str],
    table: Optional[str],
    schema: str,
    max_rows: int,
    target_column: str,
    out_path: Optional[str],
):
    """
    @brief Train and export metrics (R2/MAE/MSE/RMSE).
    """
    df = _load_dataframe(source, csv_path, table, schema, max_rows)

    model = LinearModel()
    model.train(data=df, target_column=target_column)
    metrics = model.export_metrics()

    if out_path:
        p = Path(out_path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(metrics, indent=2))
        click.echo(f"Wrote metrics: {p}")
    else:
        click.echo(json.dumps(metrics, indent=2))


@linear.command("equation")
@click.option("--source", type=click.Choice(["csv", "db"], case_sensitive=False), required=True)
@click.option("--csv", "csv_path", type=click.Path(dir_okay=False))
@click.option("--table", type=str)
@click.option("--schema", type=str, default="dbo", show_default=True)
@click.option("--max-rows", type=int, default=100_000, show_default=True)
@click.option("--target", "target_column", type=str, required=True)
@click.option("--precision", type=int, default=6, show_default=True)
def linear_equation(
    source: str,
    csv_path: Optional[str],
    table: Optional[str],
    schema: str,
    max_rows: int,
    target_column: str,
    precision: int,
):
    """
    @brief Train and print a human-readable regression equation.
    """
    df = _load_dataframe(source, csv_path, table, schema, max_rows)

    model = LinearModel()
    model.train(data=df, target_column=target_column)
    click.echo(model.export_regression_equation(precision=precision))


@linear.command("coeffs")
@click.option("--source", type=click.Choice(["csv", "db"], case_sensitive=False), required=True)
@click.option("--csv", "csv_path", type=click.Path(dir_okay=False))
@click.option("--table", type=str)
@click.option("--schema", type=str, default="dbo", show_default=True)
@click.option("--max-rows", type=int, default=100_000, show_default=True)
@click.option("--target", "target_column", type=str, required=True)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False),
    default="outputs/linear",
    show_default=True,
    help="Directory to write outputs.",
)
def linear_coeffs(
    source: str,
    csv_path: Optional[str],
    table: Optional[str],
    schema: str,
    max_rows: int,
    target_column: str,
    out_dir: str,
):
    """
    @brief Train and export coefficients to CSV.
    """
    df = _load_dataframe(source, csv_path, table, schema, max_rows)
    out = _ensure_output_dir(out_dir)

    model = LinearModel()
    model.train(data=df, target_column=target_column)
    coeffs_df = model.export_coefficients()

    coeffs_path = out / "coefficients.csv"
    coeffs_df.to_csv(coeffs_path, index=False)
    click.echo(f"Wrote coefficients: {coeffs_path}")


@linear.command("predictions")
@click.option("--source", type=click.Choice(["csv", "db"], case_sensitive=False), required=True)
@click.option("--csv", "csv_path", type=click.Path(dir_okay=False))
@click.option("--table", type=str)
@click.option("--schema", type=str, default="dbo", show_default=True)
@click.option("--max-rows", type=int, default=100_000, show_default=True)
@click.option("--target", "target_column", type=str, required=True)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False),
    default="outputs/linear",
    show_default=True,
)
def linear_predictions(
    source: str,
    csv_path: Optional[str],
    table: Optional[str],
    schema: str,
    max_rows: int,
    target_column: str,
    out_dir: str,
):
    """
    @brief Train and export predictions (actual/predicted/residual) to CSV.
    """
    df = _load_dataframe(source, csv_path, table, schema, max_rows)
    out = _ensure_output_dir(out_dir)

    model = LinearModel()
    model.train(data=df, target_column=target_column)
    preds_df = model.export_predictions()

    preds_path = out / "predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    click.echo(f"Wrote predictions: {preds_path}")


@linear.command("plot")
@click.option("--source", type=click.Choice(["csv", "db"], case_sensitive=False), required=True)
@click.option("--csv", "csv_path", type=click.Path(dir_okay=False))
@click.option("--table", type=str)
@click.option("--schema", type=str, default="dbo", show_default=True)
@click.option("--max-rows", type=int, default=100_000, show_default=True)
@click.option("--target", "target_column", type=str, required=True)
@click.option(
    "--which",
    type=click.Choice(["pred-vs-actual", "residuals", "residual-hist"], case_sensitive=False),
    required=True,
    help="Which diagnostic plot to show.",
)
def linear_plot(
    source: str,
    csv_path: Optional[str],
    table: Optional[str],
    schema: str,
    max_rows: int,
    target_column: str,
    which: str,
):
    """
    @brief Train and show diagnostic plots.
    """
    df = _load_dataframe(source, csv_path, table, schema, max_rows)

    model = LinearModel()
    model.train(data=df, target_column=target_column)

    which = which.lower()
    if which == "pred-vs-actual":
        model.plot_predicted_vs_actual()
    elif which == "residuals":
        model.plot_residuals_vs_predicted()
    elif which == "residual-hist":
        model.plot_residual_distribution()
    else:
        raise click.ClickException("Invalid plot type.")


# ----------------------------
# Entry point (optional)
# ----------------------------

if __name__ == "__main__":
    regression()

