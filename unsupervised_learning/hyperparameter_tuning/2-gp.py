#!/usr/bin/env python3
"""
Updating the Gaussian Process
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

    def predict(self, X_s):
        """
        Predicts the mean and standard deviation of points in a Gaussian
        process.

        Parameters:
        - X_s: numpy.ndarray of shape (s, 1), the points to predict

        Returns:
        - mu: numpy.ndarray of shape (s,), the mean for each point
        - sigma: numpy.ndarray of shape (s,), the variance for each point
        """
        # Precalc. the covariance matrices
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu_s = (K_s.T @ K_inv @ self.Y).reshape(-1)
        cov_s = K_ss - K_s.T @ K_inv @ K_s
        # NOTE Sigma star is the diagonal of the cov star matrix
        return mu_s, np.diag(cov_s)

    def update(self, X_new, Y_new):
        """
        Updates the Gaussian Process with a new sample point.

        Parameters:
        - X_new: numpy.ndarray of shape (1,), the new sample point.
        - Y_new: numpy.ndarray of shape (1,), the new sample function value.

        Updates the public instance attributes X, Y, and K.
        """
        # Append the new data point to existing (X, Y) points
        self.X = np.vstack((self.X, X_new.reshape(-1, 1)))
        self.Y = np.vstack((self.Y, Y_new.reshape(-1, 1)))

        # Update the covariance matrix with the new data
        self.K = self.kernel(self.X, self.X)
