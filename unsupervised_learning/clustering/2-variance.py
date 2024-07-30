#!/usr/bin/env python3
"""
Variance
"""

import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a dataset.

    Parameters:
    - X (numpy.ndarray): 2D numpy array of shape `(n, d)` with the dataset.
        - n is the number of data points
        - d is the number of dimensions for each data point
    - C (numpy.ndarray): 2D numpy array of shape `(k, d)` with the centroid
        means for each cluster.
        - k is the number of clusters
        - d is the number of dimensions for each cluster centroid

    Returns:
    - float: The total variance, or `None` on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    # Distances to closest centroid from each point
    dists = np.min(np.linalg.norm(X[:, np.newaxis] - C, axis=-1), axis=-1)

    # Variance of minimum distances to centroids (intra-cluster variance)
    return np.sum(dists ** 2)
