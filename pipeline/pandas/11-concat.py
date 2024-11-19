#!/usr/bin/env python3
"""
Concatenate two DataFrames
"""
import pandas as pd
index = __import__('10-index').index


def concat(df1: pd.DataFrame, df2: pd.DataFrame):
    """
    Concatenates two dataframes with the specified conditions.

    Parameters:
        df1 (pd.DataFrame): The first DataFrame (data from Coinbase).
        df2 (pd.DataFrame): The second DataFrame (data from Bitstamp).

    Returns:
        pd.DataFrame: The filtered, concatenated DataFrame.
    """
    # Index both DataFrames on their Timestamp column
    df1 = index(df1)
    df2 = index(df2)

    # Filter the second DataFrame up to the specified Timestamp
    df2 = df2.loc[:1417411920]

    # NOTE concatenated dataframes, using keys to differentiate data origin
    return pd.concat([df2, df1], keys=["bitstamp", "coinbase"])
