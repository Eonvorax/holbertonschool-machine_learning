#!/usr/bin/env python3
"""
Reverse sort and transpose a DataFrame
"""
import pandas as pd


def flip_switch(df: pd.DataFrame):
    """
    Sorts a DataFrame in reverse chronological order and transposes it.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The transformed DataFrame, sorted and transposed.
    """
    return df.sort_index(ascending=False).transpose()
