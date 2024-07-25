#!/usr/bin/env python3
"""
MultiNormal
"""

import numpy as np


class MultiNormal:
    """
    A class representing a Multivariate Normal distribution.
    """

    def __init__(self, data):
        """
        Initializes a MultiNormal instance.

        Parameters:
            - data (`numpy.ndarray`): The data set of shape `(d, n)`
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Calculate the means for each dimension
        self.mean = np.mean(data, axis=1).reshape(d, 1)

        # Center data: subtract the mean
        data_centered = data - self.mean

        # Using compact formula for covariance
        self.cov = np.dot(data_centered, data_centered.T) / (n - 1)

    def pdf(self, x):
        """
        Calculates the PDF at a data point.

        Parameters:
        - `x` (`numpy.ndarray`): The data point of shape `(d, 1)`

        Returns:
        - `float`: The value of the PDF at the data point `x`
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]
        if x.shape != (d, 1):
            raise ValueError(f"x must have the shape ({d}, 1)")

        # Determinant and inverse of the covariance matrix
        cov_det = np.linalg.det(self.cov)
        cov_inv = np.linalg.inv(self.cov)

        # Calc. the normalization constant
        norm_const = 1.0 / (np.sqrt((2 * np.pi) ** d * cov_det))

        # Calculate the exponent
        diff = x - self.mean
        exponent = -0.5 * (diff.T @ cov_inv @ diff)

        # Finally, the PDF value (not the scalar)
        pdf_value = norm_const * np.exp(exponent)
        return pdf_value[0, 0]
