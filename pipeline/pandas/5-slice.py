#!/usr/bin/env python3
"""
Slice every 60th row from selected DataFrame columns
"""


def slice(df):
    """
    Extracts specific columns and selects every 60th row from a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The sliced DataFrame containing the selected columns
            and every 60th row.
    """
    return df[["High", "Low", "Close", "Volume_(BTC)"]].iloc[::60]
