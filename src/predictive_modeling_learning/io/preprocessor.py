from dataclasses import dataclass

import pandas as pd


@dataclass
class DataSummary:
    shape: tuple[int, int]
    dtypes: dict[str, str]
    null_counts: dict[str, int]
    null_percentages: dict[str, float]
    stats: pd.DataFrame  # output of df.describe()


# ---- Data Inspection ----
def inspect(df: pd.DataFrame) -> DataSummary:
    """
    Inspects the DataFrame and returns a summary of its structure and contents.
    Args:
        df (pd.DataFrame): The DataFrame to inspect.
    Returns:
        DataSummary: A dataclass containing the summary of the DataFrame.
            Contains the following attributes:
            - shape: A tuple representing the number of rows and columns in the
                DataFrame.
            - dtypes: A dictionary mapping column names to their data types.
            - null_counts: A dictionary mapping column names to the count of null values
                in each column.
            - null_percentages: A dictionary mapping column names to the percentage of
                null values.
            - stats: A DataFrame containing summary statistics for numeric columns
                (output of df.describe()).
    """
    return DataSummary(
        shape= (df.shape[0], df.shape[1]),
        dtypes= df.dtypes.astype(str).to_dict(),
        null_counts= df.isnull().sum().to_dict(),
        null_percentages= (df.isnull().mean() * 100).to_dict(),
        stats= df.describe(),
    )


# ---- Data Cleaning ----
def drop_null(df: pd.DataFrame, threshold: float = 0.5, axis: int = 1) -> pd.DataFrame: 
    """
    Drops columns from the DataFrame that have a percentage of null values above the
    specified threshold.
    Args:
        df (pd.DataFrame): The DataFrame to process.
        threshold (float): The percentage threshold for dropping columns. Columns with
            a percentage of null values above this threshold will be dropped.
            Default is 0.5 (50%).
    Returns:
        pd.DataFrame: A new DataFrame with columns dropped based on the null value
            threshold.
    """
    null_percentages = df.isnull().mean(axis=0 if axis == 1 else 1)
    labels_to_drop = null_percentages[null_percentages > threshold].index
    return df.drop(labels=labels_to_drop, axis=axis)


def fill_null(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """
    Fills null values in the DataFrame using the specified strategy.
    Args:
        df (pd.DataFrame): The DataFrame to process.
        strategy (str): The strategy for filling null values. Options are:
            - "mean": Fill with the mean of the column (numeric columns only).
            - "median": Fill with the median of the column (numeric columns only).
            - "mode": Fill with the mode of the column (categorical columns only).
            Default is "mean".
    Returns:
        pd.DataFrame: A new DataFrame with null values filled based on the specified
            strategy.
    """
    if strategy == "mean":
        numeric_fill = df.mean(numeric_only=True)
        cat_fill = df.select_dtypes(exclude="number").mode().iloc[0]
        return df.fillna({**numeric_fill, **cat_fill})
    elif strategy == "median":
        return df.fillna(df.median(numeric_only=True))
    elif strategy == "mode":
        return df.fillna(df.mode().iloc[0])
    else:
        raise ValueError(
            f"Invalid strategy: {strategy}. Choose from 'mean', 'median', or 'mode'."
        )


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops duplicate rows from the DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame to process.
    Returns:
        pd.DataFrame: A new DataFrame with duplicate rows removed.
    """
    return df.drop_duplicates()


def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Drops specified columns from the DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame to process.
        columns (list[str]): A list of column names to drop from the DataFrame.
    Returns:
        pd.DataFrame: A new DataFrame with the specified columns removed.
    """
    return df.drop(columns=columns)


# ---- Type Handling ----
def infer_types(df: pd.DataFrame) -> dict[str, str]:
    """
    Infers the semantic type of each column in the DataFrame as 'numerical', 'categorical', or 'datetime'.
    Args:
        df (pd.DataFrame): The DataFrame to process.
    Returns:
        dict[str, str]: A dictionary mapping column names to their inferred semantic type.
    """
    types = {}
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            types[col] = "numerical"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            types[col] = "datetime"
        else:
            types[col] = "categorical"
    return types


def convert_types(df: pd.DataFrame, type_mapping: dict[str, str]) -> pd.DataFrame:
    """
    Converts the data types of specified columns in the DataFrame based on a provided
    mapping.
    Args:
        df (pd.DataFrame): The DataFrame to process.
        type_mapping (dict[str, str]): A dictionary mapping column names to their
            desired data types (e.g., {"column_name": "int", "another_column": "float"}).
    Returns:
        pd.DataFrame: A new DataFrame with the specified columns converted to the
            desired data types.
    """
    missing = [col for col in type_mapping if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")
    return df.astype(type_mapping)