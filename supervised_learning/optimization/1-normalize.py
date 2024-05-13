#!/usr/bin/env python3
"""
Normalizes a matrix using its means and std deviation.
"""

import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix.

    Args:
        X: numpy.ndarray of shape (d, nx) to normalize
           d: number of data points
           nx: number of features
        m: numpy.ndarray of shape (nx,) that contains the mean of all
            features of X
        s: numpy.ndarray of shape (nx,) that contains the standard deviation
            of all features of X

    Returns:
        The normalized X matrix.
    """
    # NOTE using numpy built-in element-wise operations
    return (X - m) / s
