#!/usr/bin/env python3
"""
Set column as index
"""


def index(df):
    """
    Sets the 'Timestamp' column as the index of the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The modified DataFrame with 'Timestamp' as the index.
    """
    if "Timestamp" in df.columns:
        df = df.set_index(["Timestamp"])
    return df
