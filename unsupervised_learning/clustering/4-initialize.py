#!/usr/bin/env python3
"""
Initialize GMM
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model.

    Parameters:
    X (numpy.ndarray): 2D numpy array of shape (n, d) containing the dataset.
    k (int): A positive integer containing the number of clusters.

    Returns:
    tuple: (pi, m, S), or (None, None, None) on failure.
        - pi is a numpy.ndarray of shape (k,) containing the priors for each
        cluster, initialized evenly.
        - m is a numpy.ndarray of shape (k, d) containing the centroid means
        for each cluster, initialized with K-means.
        - S is a numpy.ndarray of shape (k, d, d) containing the covariance
        matrices for each cluster, initialized as identity matrices.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None

    # Initialize equal prior probabilities with each cluster
    pi = np.full((k,), fill_value=1/k)

    # m (centroids) is initialized using K-means
    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None

    # S is initialized with identity matrices
    S = np.tile(np.eye(X.shape[1]), (k, 1, 1))

    return pi, m, S
