#!/usr/bin/env python3
"""
Mean and Covariance
"""

import numpy as np


def mean_cov(X):
    """
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape

    if n < 2:
        raise ValueError("X must contain multiple data points")

    means = np.mean(X, axis=0)
    cov = []
    for i in range(n):
        for j in range(d):
            cov[i][j] = # TODO
    return np.mean(X, axis=0), cov
