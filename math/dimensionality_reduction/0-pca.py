#!/usr/bin/env python3
"""
PCA
"""

import numpy as np


def pca(X, var=0.95):
    """
    Perform PCA on the dataset X to retain a fraction var of the variance.

    Parameters:
    - X (numpy.ndarray): The input data matrix of shape (n, d) with zero mean.
    - var (float): The fraction of the variance to be retained by the
    PCA transformation.

    Returns:
    - numpy.ndarray: The weights matrix W of shape (d, nd) where nd is the
    new dimensionality.
    """
    # Compute the SVD of the data matrix
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Compute the total variance explained by the singular values
    cum_variance = np.cumsum(S ** 2) / np.sum(S ** 2)

    # Determine the number of components to keep (indexing starts at 0)
    num_components = np.argmax(cum_variance >= var) + 1

    # Transposed (for shape(d, nd)) top "num_components + 1" rows of Vt
    return Vt[:num_components + 1].T
