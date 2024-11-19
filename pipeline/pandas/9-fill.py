#!/usr/bin/env python3
"""
Fill missing values
"""


def fill(df):
    """
    Cleans and fills missing values in the DataFrame :
    - Fills missing values in the Close column with the previous row's value.
    - Fills missing values in the High, Low, and Open columns with the
    corresponding Close value in the same row.
    - Sets missing values in Volume_(BTC) and Volume_(Currency) to 0.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with missing values handled as described
    """
    # Remove the "Weighted_Price" column
    if "Weighted_Price" in df.columns:
        df = df.drop(columns=["Weighted_Price"])

    # Fill missing values in the "Close" column with the previous row's value
    if "Close" in df.columns:
        df["Close"] = df["Close"].ffill()

    # Fill missing values in these columns with corresponding "Close" values
    for column in ["High", "Low", "Open"]:
        if column in df.columns:
            df[column] = df[column].fillna(df["Close"])

    # Set missing values in the Volume columns to 0
    for column in ["Volume_(BTC)", "Volume_(Currency)"]:
        if column in df.columns:
            df[column] = df[column].fillna(0)

    return df
