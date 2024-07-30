#!/usr/bin/env python3
"""
Optimizing k - Kmeans
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance.

    Parameters:
    - X (numpy.ndarray): 2D numpy array of shape (n, d) containing the dataset.
    - kmin (int): Minimum number of clusters to check for (inclusive).
    - kmax (int): Maximum number of clusters to check for (inclusive).
    - iterations (int): Maximum number of iterations for K-means.

    Returns:
    - tuple: (results, d_vars), or (None, None) on failure.
        - results is a list containing the outputs of K-means for each
        cluster size.
        - d_vars is a list containing the difference in variance from the
        smallest cluster size for each cluster size.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax < kmin):
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    if isinstance(kmax, int) and kmax <= kmin:
        return None, None

    if kmax is None:
        max_clusters = X.shape[0]
    else:
        max_clusters = kmax

    results = []
    d_vars = []

    # Calculations with kmin (smallest cluster size)
    C, clss = kmeans(X, kmin, iterations)
    base_variance = variance(X, C)
    results.append((C, clss))
    # Base difference with first variance (with itself) is zero
    d_vars = [0.0]

    # With each following cluster size k:
    k = kmin + 1
    while k < max_clusters + 1:
        # Run kmeans algorithm and calc. variance of distance to centroids
        C, clss = kmeans(X, k, iterations)
        current_variance = variance(X, C)
        # Add results and variances differences to the lists
        results.append((C, clss))
        d_vars.append(base_variance - current_variance)
        k += 1

    return results, d_vars
