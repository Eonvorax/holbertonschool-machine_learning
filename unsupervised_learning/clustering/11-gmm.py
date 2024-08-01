#!/usr/bin/env python3
"""
GMM calculation with sklearn
"""

import sklearn.mixture


def gmm(X, k):
    """
    Calculates a Gaussian Mixture Model from a dataset using the specified
    number of clusters.

    Parameters:
    - X (numpy.ndarray): The dataset with shape (n, d), where n is the number
    of samples and d is the number of features.
    - k (int): The number of clusters.

    Returns:
    - pi (numpy.ndarray): A numpy array of shape (k,) containing the cluster
    priors.
    - m (numpy.ndarray): A numpy array of shape (k, d) containing the centroid
    means.
    - S (numpy.ndarray): A numpy array of shape (k, d, d) containing the
    covariance matrices.
    - clss (numpy.ndarray): A numpy array of shape (n,) containing the cluster
    indices for each data point.
    - bic (float): The Bayesian Information Criterion value for the model.
    """
    model = sklearn.mixture.GaussianMixture(n_components=k).fit(X)

    pi = model.weights_
    m = model.means_
    S = model.covariances_
    clss = model.predict(X)
    bic = model.bic(X)

    return pi, m, S, clss, bic
