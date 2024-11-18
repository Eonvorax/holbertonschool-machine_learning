#!/usr/bin/env python3
"""
Rename column and convert timestamp values
"""
import pandas as pd


def rename(df: pd.DataFrame):
    """
    Renames the Timestamp column of the given DataFrame to Datetime, and
    converts the timestamp values to datatime values. It then returns only
    the Datetime and Close columns.

    :param pd.DataFrame: the input DataFrame

    Returns:
        the modified `pd.DataFrame`, with only the Datetime and Close columns
    """
    # Rename the 'Timestamp' column to 'Datetime' using mapping dict
    df = df.rename(columns={"Timestamp": "Datetime"})

    # Convert the renamed column to the datetime format
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit='s')

    # Return only the required columns
    return df[["Datetime", "Close"]]
