#!/usr/bin/env python3
"""
Correlation
"""

import numpy as np


def correlation(C):
    """
    Calculate the correlation matrix from a covariance matrix.

    Parameters:
    C (numpy.ndarray): The covariance matrix of shape (d, d)

    Returns:
    numpy.ndarray: The correlation matrix of shape (d, d)
    """
    # Check if C is a numpy array
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    # Check if C is a 2D square matrix
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    d = C.shape[0]

    # NOTE found a (nice?) trick: corr = D-inv Σ D-inv
    # Calculate standard deviations (sqrt of diagonal elements)
    std_devs = np.sqrt(np.diag(C))

    # Calc. inverse of the diagonal matrix of standard deviations
    D_inv = np.diag(1 / std_devs)

    return D_inv @ C @ D_inv
