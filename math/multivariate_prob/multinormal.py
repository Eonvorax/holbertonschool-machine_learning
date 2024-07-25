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
