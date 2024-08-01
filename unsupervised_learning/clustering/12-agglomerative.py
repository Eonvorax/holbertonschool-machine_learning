#!/usr/bin/env python3
"""
Agglomerative clustering
"""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Performs agglomerative clustering on a dataset and displays the resulting
    dendrogram with each cluster displayed in a different color.

    Parameters:
    - X (numpy.ndarray): The dataset with shape (n, d), where n is the number
    of samples and d is the number of features.
    - dist (float): The maximum cophenetic distance for all clusters.

    Returns:
    - clss (numpy.ndarray): A numpy array of shape (n,) containing the cluster
    indices for each data point.
    """
    # Perform hierarchical/agglomerative clustering using Ward linkage
    linkage_matrix = scipy.cluster.hierarchy.linkage(X, method="ward")

    # Display the dendrogram
    dendrogram = scipy.cluster.hierarchy.dendrogram(
        linkage_matrix, color_threshold=dist)

    # # Cleaner labeling
    # plt.title('Dendrogram')
    # plt.xlabel('Data Points')
    # plt.ylabel('Euclidean Distance')
    plt.show()

    # Return clusters based on the given distance threshold
    return scipy.cluster.hierarchy.fcluster(Z=linkage_matrix,
                                            t=dist, criterion="distance")
