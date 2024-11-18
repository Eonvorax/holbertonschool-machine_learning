#!/usr/bin/env python3
"""
Rename column and convert timestamp values
"""
import pandas as pd


def array(df: pd.DataFrame):
    """
    Extracts the last 10 rows of the 'High' and 'Close' columns from a
    DataFrame and converts them into a numpy.ndarray.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing 'High' and
            'Close' columns.

    Returns:
        np.ndarray: A numpy array of the last 10 rows of 'High' and 'Close'
            columns.
    """
    # Last 10 rows of 'High' and 'Close' columns, converted to a numpy array
    return df[["High", "Close"]].tail(10).to_numpy()
