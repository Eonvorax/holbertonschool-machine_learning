#!/usr/bin/env python3
"""
Compute descriptive stats
"""


def analyze(df):
    """
    Computes descriptive statistics for all columns except the Timestamp
    column.

    Parameters:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        pd.DataFrame: A DataFrame containing the descriptive statistics.
    """
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])
    return df.describe()
