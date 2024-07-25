#!/usr/bin/env python3
"""
Mean and Covariance
"""

import numpy as np


def mean_cov(X):
    """
    Calculate the mean and covariance of a data set.

    Parameters:
    X (numpy.ndarray): The data set of shape (n, d)

    Returns:
    tuple: (mean, cov)
        - mean (numpy.ndarray): Shape (1, d) containing the mean of
        the data set
        - cov (numpy.ndarray): Shape (d, d) containing the covariance
        matrix of the data set
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape

    if n < 2:
        raise ValueError("X must contain multiple data points")

    # Calculate the means for each dimension
    means = np.mean(X, axis=0).reshape(1, d)

    # Center data: subtract the mean
    X_centered = X - means

    # Using compact formula for covariance
    cov = np.dot(X_centered.T, X_centered) / (n - 1)
    return means, cov
