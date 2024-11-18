#!/usr/bin/env python3
"""
Remove entries with NaN values
"""


def prune(df):
    """
    Removes any rows from a DataFrame where the 'Close' column has NaN values.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        (pd.DataFrame): The DataFrame with rows containing NaN in the 'Close'
        column removed.
    """
    return df.dropna(subset=["Close"])
