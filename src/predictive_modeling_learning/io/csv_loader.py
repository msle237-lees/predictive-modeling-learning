from pathlib import Path

import pandas as pd


def load_csv(
    file_path: str,
    delimiter: str = ",",
    encoding: str = "utf-8",
    header: int | None = 0,
) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    path = Path(file_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"No file found at: {path}")
    if path.suffix != ".csv":
        raise ValueError(f"Expected a .csv file, got: {path.suffix}")

    return pd.read_csv(path, delimiter=delimiter, encoding=encoding, header=header)
