#!/usr/bin/env python3
"""
Kmeans with sklearn
"""

import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means clustering on the dataset X with k clusters.

    Parameters:
    - X (numpy.ndarray): The dataset to cluster, with shape (n, d),
    where n is the number of samples and d is the number of features.
    - k (int): The number of clusters to form.

    Returns:
    - C (numpy.ndarray): A numpy array of shape (k, d) containing the
    centroid means for each cluster.
    - clss (numpy.ndarray): A numpy array of shape (n,) containing the
    index of the cluster in C that each data point belongs to.

    """
    kmeans_result = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    return kmeans_result.cluster_centers_, kmeans_result.labels_
