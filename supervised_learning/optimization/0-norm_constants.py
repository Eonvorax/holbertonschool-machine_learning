#!/usr/bin/env python3
"""
Calculates the normalization (standardization) constants of a matrix.
"""

import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization (standardization) constants of a matrix.

    Args:
        X: numpy.ndarray of shape (m, nx) to normalize
           m: number of data points
           nx: number of features

    Returns:
        Tuple containing the mean and standard deviation of each feature.
    """
    # NOTE calculating along axis 0, collapsing the rows in a single value
    # i.e. for each feature column
    return np.mean(X, axis=0), np.std(X, axis=0)
