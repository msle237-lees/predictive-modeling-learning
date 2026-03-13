import pandas as pd

from .csv_loader import load_csv
from .db_loader import connect, query


def load(source: str, **kwargs) -> pd.DataFrame:
    if source == "csv":
        return load_csv(**kwargs)
    elif source == "db":
        engine = connect(kwargs.pop("connection_string"))
        return query(engine, kwargs.pop("sql"))
    else:
        raise ValueError(f"Unknown source: {source}")
