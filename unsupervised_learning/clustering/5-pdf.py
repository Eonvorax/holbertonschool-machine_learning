#!/usr/bin/env python3
"""
PDF of a Gaussian distribution
"""

import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution.

    Parameters:
    - X (numpy.ndarray): A 2D numpy array of shape (n, d) containing the data
    points whose PDF should be evaluated.
    - m (numpy.ndarray): A 1D numpy array of shape (d,) containing the mean of
    the distribution.
    - S (numpy.ndarray): A 2D numpy array of shape (d, d) containing the
    covariance of the distribution.

    Returns:
    - numpy.ndarray: A 1D numpy array of shape (n,) containing the PDF values
    for each data point, or None on failure. All values in the result should
    have a minimum value of 1e-300.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(m, np.ndarray) or m.ndim != 1:
        return None
    if not isinstance(S, np.ndarray) or S.ndim != 2:
        return None
    if (X.shape[1] != m.shape[0] or S.shape[0] != S.shape[1]
            or S.shape[0] != m.shape[0]):
        return None

    n, d = X.shape

    # Determinant and inverse of the covariance matrix
    det_S = np.linalg.det(S)
    if det_S == 0:
        # Determinant is zero, can't invert, stopping here
        return None

    inv_S = np.linalg.inv(S)

    # Calc. the normalization factor
    norm_const = 1.0 / (np.sqrt((2 * np.pi) ** d * det_S))

    # Calculate the exponent, using diff. between mean and data values
    diff = X - m
    exponent = -0.5 * np.sum(diff @ inv_S * diff, axis=1)
    # NOTE careful with the matmul/mul operators above

    # Finally, the PDF values, with no value less than 1e-300
    pdf_values = norm_const * np.exp(exponent)
    return np.maximum(pdf_values, 1e-300)
