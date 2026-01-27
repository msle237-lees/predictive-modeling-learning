"""
./src/predictive_modeling_learning/io/db.py
Database input/output utilities for predictive modeling learning.

Uses requests to interact with a remote database API.

Author: Michael Lees
Date: 2026-01-22
"""

import os
import json
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")


def _require_base_url() -> str:
    """
    @brief Ensure API_BASE_URL is set.

    @return API base url string.
    @throws RuntimeError if API_BASE_URL is missing.
    """
    if not API_BASE_URL or not API_BASE_URL.strip():
        raise RuntimeError("API_BASE_URL is not set. Please define it in your environment or .env file.")
    return API_BASE_URL.rstrip("/")


def get_tables() -> Dict[str, Any]:
    """
    @brief Fetch the list of available tables from the database API.

    @return JSON response as dict.
    """
    base = _require_base_url()
    response = requests.get(f"{base}/tables", timeout=30)
    response.raise_for_status()
    return response.json()


def drop_table(table_name: str, schema: str = "dbo") -> Dict[str, Any]:
    """
    @brief Drop a specific table in the database API.

    @param table_name Table to drop.
    @param schema Schema name (default: dbo).
    @return JSON response as dict.
    """
    base = _require_base_url()
    payload = {"schema": schema, "table": table_name, "if_exists": True}
    response = requests.post(f"{base}/tables/drop", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def fetch_table_data(
    table_name: str,
    schema: str = "dbo",
    columns: Optional[List[str]] = None,
    where: Optional[Dict[str, Any]] = None,
    batch_size: int = 5000,
    max_rows: int = 100_000,
) -> pd.DataFrame:
    """
    @brief Fetch *all* data from a specific table in the database API.

    This uses the API endpoint:
      POST /rows/select-all

    @param table_name Target table name.
    @param schema Schema name (default: dbo).
    @param columns Optional list of columns to return (default: all).
    @param where Optional equality filters {col: value}.
    @param batch_size Rows fetched per internal batch on the server.
    @param max_rows Safety cap for total rows returned.
    @return DataFrame containing the selected rows.
    """
    base = _require_base_url()

    payload = {
        "schema": schema,
        "table": table_name,
        "columns": columns,
        "where": where,
        "batch_size": batch_size,
        "max_rows": max_rows,
    }

    response = requests.post(f"{base}/rows/select-all", json=payload, timeout=300)
    response.raise_for_status()

    # Expected shape from your API: GenericResult { ok, message, data, rows_affected }
    result = response.json()

    if isinstance(result, dict) and result.get("ok") is False:
        raise RuntimeError(f"API returned ok=false: {result.get('message')}")

    data = result.get("data") if isinstance(result, dict) else result

    if data is None:
        return pd.DataFrame()

    if not isinstance(data, list):
        # If API ever returns a dict or something unexpected, normalize.
        return pd.DataFrame([data])

    return pd.DataFrame(data)


def create_table(table_name: str, schema: Dict[str, str]) -> Dict[str, Any]:
    """
    @brief Create a new table in the database API.

    Args:
        table_name: Name of the table to create
        schema: Dict mapping column names to types (e.g., {"id": "int", "value": "float"})
    """
    base = _require_base_url()

    # Convert schema dict to columns list format expected by your API
    columns = []
    for col_name, col_type in schema.items():
        columns.append(
            {
                "name": col_name,
                "type": col_type,
                "nullable": True,
                "identity": False,
                "primary_key": False,
                "default": None,
            }
        )

    payload = {
        "schema": "dbo",
        "table": table_name,
        "if_not_exists": True,
        "columns": columns,
    }

    response = requests.post(f"{base}/tables/create", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def create_analysis_table(table_name: str, schema: Dict[str, str], data: pd.DataFrame) -> Dict[str, Any]:
    """
    @brief Create a new analysis table in the database API.

    Args:
        table_name: Name of the table to create
        schema: Dict mapping column names to types (e.g., {"id": "int", "value": "float"})
        data: pandas DataFrame with the data to structure the table
    """
    base = _require_base_url()

    # Convert schema dict to columns list format expected by your API
    columns = []
    for col_name, col_type in schema.items():
        columns.append(
            {
                "name": col_name,
                "type": col_type,
                "nullable": True,
                "identity": False,
                "primary_key": False,
                "default": None,
            }
        )

    payload = {
        "schema": "dbo",
        "table": table_name,
        "if_not_exists": True,
        "columns": columns,
    }

    response = requests.post(f"{base}/tables/create", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()

if __name__ == "__main__":
    tables = get_tables()

    print("Available tables:")
    print(json.dumps(tables, indent=2))

    # Example: Fetch ALL rows from any table
    df = fetch_table_data("sensor_table", max_rows=100000)
    print("Fetched table data:")
    print(df.head())
    print(f"\nRows: {len(df)}  Cols: {len(df.columns)}")
