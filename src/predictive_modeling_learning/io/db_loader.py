import pandas as pd
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine


def connect(connection_string: str) -> Engine:
    """Create a database engine from a connection string."""
    if not connection_string:
        raise ValueError("Connection string cannot be empty")

    try:
        engine = create_engine(connection_string)
        # Test the connection immediately rather than failing later
        with engine.connect() as _:
            pass
        return engine
    except Exception as e:
        raise ConnectionError(f"Failed to connect to database: {e}")


def query(engine: Engine, sql: str) -> pd.DataFrame:
    """Execute a SQL query and return results as a DataFrame."""
    if not sql.strip():
        raise ValueError("SQL query cannot be empty")

    try:
        with engine.connect() as conn:
            return pd.read_sql(sql, conn)  # passes the open connection through
    except Exception as e:
        raise RuntimeError(f"Query failed: {e}")


def list_tables(engine: Engine) -> list[str]:
    """Return a list of all table names in the database."""
    try:
        inspector = inspect(engine)
        return inspector.get_table_names()
    except Exception as e:
        raise RuntimeError(f"Failed to list tables: {e}")


def inspect_table(engine: Engine, table_name: str) -> pd.DataFrame:
    """Return column names, types, and row count for a given table."""
    try:
        inspector = inspect(engine)

        # Validate table exists before querying
        if table_name not in inspector.get_table_names():
            raise ValueError(f"Table '{table_name}' does not exist")

        # Get column metadata
        columns = inspector.get_columns(table_name)
        col_info = pd.DataFrame(
            [{"column": col["name"], "dtype": str(col["type"])} for col in columns]
        )

        # Get row count
        with engine.connect() as conn:
            count = pd.read_sql(f"SELECT COUNT(*) as row_count FROM {table_name}", conn)

        col_info["row_count"] = count["row_count"].iloc[0]
        return col_info
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to inspect table '{table_name}': {e}")
