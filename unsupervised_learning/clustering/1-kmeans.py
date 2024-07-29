#!/usr/bin/env python3

"""
Kmeans
"""

import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means clustering.

    Parameters:
    X (numpy.ndarray): A 2D numpy array of shape (n, d) containing the dataset
                       that will be used for K-means clustering.
                       - n is the number of data points
                       - d is the number of dimensions for each data point
    k (int): A positive integer representing the number of clusters.

    Returns:
    numpy.ndarray: A 2D numpy array of shape (k, d) containing the initialized
                   centroids for each cluster.
                   Returns None on failure
    """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    low_values = np.min(X, axis=0)
    high_values = np.max(X, axis=0)

    return np.random.uniform(low_values, high_values, size=(k, X.shape[1]))


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset.

    Parameters:
    X (numpy.ndarray): A 2D numpy array of shape (n, d) containing the dataset.
                       - n is the number of data points
                       - d is the number of dimensions for each data point
    k (int): A positive integer representing the number of clusters.
    iterations (int): A positive integer representing the maximum number of
                      iterations that should be performed.

    Returns:
    tuple: (C, clss) on success, or (None, None) on failure.
           - C is a numpy.ndarray of shape (k, d) containing the centroid means
             for each cluster.
           - clss is a numpy.ndarray of shape (n,) containing the index of the
             cluster in C that each data point belongs to.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    ctds = initialize(X, k)
    if ctds is None:
        return None, None

    for _ in range(iterations):
        prev_ctds = np.copy(ctds)

        # Calculate distances and assign clusters
        dists = np.sqrt(np.sum((X - ctds[:, np.newaxis]) ** 2, axis=2))
        clss = np.argmin(dists, axis=0)

        for i in range(k):
            # Mask: points present in cluster
            cluster_mask = X[clss == i]
            if len(cluster_mask) == 0:
                ctds[i] = initialize(X, 1)
            else:
                ctds[i] = np.mean(X[clss == i], axis=0)

        # Recalculate distances and reassign clusters
        dists = np.sqrt(np.sum((X - ctds[:, np.newaxis]) ** 2, axis=2))
        clss = np.argmin(dists, axis=0)

        # Convergence check (if points haven't changed clusters)
        if np.allclose(ctds, prev_ctds):
            break

    return ctds, clss
