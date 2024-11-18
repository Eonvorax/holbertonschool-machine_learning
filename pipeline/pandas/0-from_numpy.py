#!/usr/bin/env python3
"""
Dataframe from numpy ndarray
"""
import pandas as pd


def from_numpy(array):
    """
    Creates a Pandas DataFrame from a NumPy ndarray.
    The columns of the pd.DataFrame are labeled in alphabetical order
    and capitalized. We will assume there is no more than 26 columns.

    Parameters:
        array (np.ndarray): source array to use

    Returns:
        pd.DataFrame: the resulting DataFrame
    """
    # Generate a list of column labels as capitalized alphabetical letters
    num_columns = array.shape[1]
    column_labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    return pd.DataFrame(array, columns=list(column_labels[:num_columns]))
