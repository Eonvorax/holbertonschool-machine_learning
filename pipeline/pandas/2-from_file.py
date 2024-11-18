#!/usr/bin/env python3
"""
Dataframe from .csv file
"""
import pandas as pd


def from_file(filename, delimiter):
    """
    Loads a DataFrame from a .csv file and returns it.

    Parameters:
        filename (str): name of the file to load from
        delimiter (str): the column separator

    Returns:
        pd.Dataframe: the loaded DataFrame
    """
    return pd.read_csv(filename, delimiter=delimiter)
