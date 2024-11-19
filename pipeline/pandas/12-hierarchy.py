#!/usr/bin/env python3
"""
Concatenate two DataFrames
"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Concatenates two dataframes with the specified conditions.
    Rearranges the MultiIndex so that Timestamp is the first level.
    Adds keys to the data, labeling rows from df2 as bitstamp and rows
    from df1 as coinbase.
    Ensures the data is displayed in chronological order.

    Parameters:
        df1 (pd.DataFrame): The first DataFrame (data from Coinbase).
        df2 (pd.DataFrame): The second DataFrame (data from Bitstamp).

    Returns:
        pd.DataFrame: The filtered, concatenated DataFrame.
    """

    # Index both DataFrames on their Timestamp column
    df1 = index(df1)
    df2 = index(df2)

    # Filter the DataFrames for the specified Timestamp range
    df1 = df1.loc[1417411980:1417417980]
    df2 = df2.loc[1417411980:1417417980]

    # NOTE concatenated dataframes, using keys to differentiate data origin
    df = pd.concat([df2, df1], keys=["bitstamp", "coinbase"])

    # Reorder index levels to make Timestamp the topmost (leftmost) level
    df = df.reorder_levels([1, 0], axis=0)

    # Sort by chronological order
    return df.sort_index()
