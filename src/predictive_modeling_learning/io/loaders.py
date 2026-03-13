import pandas as pd

from .csv_loader import load_csv
from .db_loader import connect, query


def load(source: str, **kwargs) -> pd.DataFrame:
    """Unified entry point for loading data from various sources."""
    if source == "csv":
        return load_csv(**kwargs)
    elif source == "db":
        return query(**kwargs)
    else:
        raise ValueError(f"Unknown source: {source}")
