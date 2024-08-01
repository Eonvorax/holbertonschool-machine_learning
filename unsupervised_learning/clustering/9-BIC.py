#!/usr/bin/env python3
"""
Bayesian Information Criterion w/ GMMs
"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a Gaussian Mixture Model using the
    Bayesian Information Criterion (BIC).

    Parameters:
    - X (numpy.ndarray): The dataset to be analyzed, of shape (n, d) where n
    is the number of data points and d is the number of dimensions.
    - kmin (int, optional): The minimum number of clusters to check
    (inclusive). Default is 1.
    - kmax (int, optional): The maximum number of clusters to check
    (inclusive). If None, it is set to the maximum number of clusters possible.
    - iterations (int, optional): The maximum number of iterations for the
    Expectation-Maximization (EM) algorithm. Default is 1000.
    - tol (float, optional): The tolerance for convergence of the EM algorithm.
    Default is 1e-5.
    - verbose (bool, optional): If True, prints log likelihood information
    during the EM algorithm execution. Default is False.

    Returns:
    - best_k (int): The optimal number of clusters based on BIC.
    - best_result (tuple): A tuple containing:
        - pi (numpy.ndarray): The priors for each cluster for the best number
        of clusters, of shape (k,).
        - m (numpy.ndarray): The centroid means for each cluster for the best
        number of clusters, of shape (k, d).
        - S (numpy.ndarray): The covariance matrices for each cluster for the
        best number of clusters, of shape (k, d, d).
    - likelihoods (numpy.ndarray): The log likelihoods for each cluster size
    tested, of shape (kmax - kmin + 1).
    - b (numpy.ndarray): The BIC values for each cluster size tested, of
    shape (kmax - kmin + 1).

    Returns `(None, None, None, None)` on failure.
    """
    if (
        not isinstance(X, np.ndarray) or X.ndim != 2
        or not isinstance(kmin, int) or kmin <= 0
        or kmax is not None and (not isinstance(kmax, int) or kmax < kmin)
        or not isinstance(iterations, int) or iterations <= 0
        or isinstance(kmax, int) and kmax <= kmin
        or not isinstance(iterations, int) or iterations <= 0
        or not isinstance(tol, float) or tol < 0
        or not isinstance(verbose, bool)
    ):
        return None, None, None, None

    n, d = X.shape
    if kmax is None:
        # Undefined, set to maximum possible
        kmax = n
    if not isinstance(kmax, int) or kmax < 1 or kmax < kmin or kmax > n:
        return None, None, None, None

    b = []
    likelihoods = []

    # With each cluster size from kmin to kmax
    for k in range(kmin, kmax + 1):
        # Find the best fit with the GMM and current cluster size k
        pi, m, S, g, li = expectation_maximization(
            X, k, iterations, tol, verbose)

        if pi is None or m is None or S is None or g is None:
            return None, None, None, None
        # NOTE p is the number of parameters, so k * d with the means,
        # k * d * (d + 1) with the covariance matrix, and k - 1 with the priors
        p = (k * d) + (k * d * (d + 1) // 2) + (k - 1)
        bic = p * np.log(n) - 2 * li

        # Save log likelihood and BIC value with current cluster size
        likelihoods.append(li)
        b.append(bic)

        # Compare current BIC to best observed BIC
        if k == kmin or bic < best_bic:
            # Update the return values
            best_bic = bic
            best_results = (pi, m, S)
            best_k = k

    likelihoods = np.array(likelihoods)
    b = np.array(b)
    return best_k, best_results, likelihoods, b
