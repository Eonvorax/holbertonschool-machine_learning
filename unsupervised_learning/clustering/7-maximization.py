#!/usr/bin/env python3
"""
Maximization step, EM algorithm with GMM
"""

import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM.

    Parameters:
    - X (numpy.ndarray): 2D numpy array of shape (n, d) containing the dataset.
    - g (numpy.ndarray): 2D numpy array of shape (k, n) containing the
    posterior probabilities for each data point in each cluster.

    Returns:
    - pi (numpy.ndarray): 1D numpy array of shape (k,) containing the updated
    priors for each cluster.
    - m (numpy.ndarray): 2D numpy array of shape (k, d) containing the updated
    centroid means for each cluster.
    - S (numpy.ndarray): 3D numpy array of shape (k, d, d) containing the
    updated covariance matrices for each cluster.
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
        not isinstance(g, np.ndarray) or g.ndim != 2 or
        X.shape[0] != g.shape[1] or
            not np.allclose(g.sum(axis=0), 1.0)):
        return None, None, None

    n, d = X.shape
    k, _ = g.shape

    # Update the priors
    pi = np.sum(g, axis=1) / n

    # Update the centroids
    m = np.dot(g, X) / np.sum(g, axis=1)[:, np.newaxis]

    # Update the covariance matrices, using the new centroids
    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]
        S[i] = np.dot(g[i] * diff.T, diff) / np.sum(g[i])

    return pi, m, S
