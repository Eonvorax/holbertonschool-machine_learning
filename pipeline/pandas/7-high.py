#!/usr/bin/env python3
"""
Sort a DataFrame by Column value
"""


def high(df):
    """
    Sorts a DataFrame by the 'High' column in descending order.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame sorted by the 'High' column in
            descending order.
    """
    return df.sort_values(by=["High"], ascending=False)
