#!/usr/bin/env python3
"""
Initialize Gaussian Process
"""

import numpy as np


class GaussianProcess:
    """
    This class represents a noiseless 1D Gaussian process.
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Initializes the Gaussian Process.

        Parameters:
        - X_init: numpy.ndarray of shape (t, 1), the inputs already sampled.
        - Y_init: numpy.ndarray of shape (t, 1), the outputs for each input.
        - l: float, the length parameter for the kernel.
        - sigma_f: float, the standard deviation of the output.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        # Compute covariance kernel matrix for the initial data
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Calculates the RBF kernel (covariance matrix) between two matrices.

        Parameters:
        - X1: numpy.ndarray of shape (m, 1).
        - X2: numpy.ndarray of shape (n, 1).

        Returns:
        - Covariance kernel matrix as a numpy.ndarray of shape (m, n).
        """
        # NOTE Squared Euclidean distance between each pair of points
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + \
            np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        # Calculate the RBF kernel matrix
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)
