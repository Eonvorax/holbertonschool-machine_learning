#!/usr/bin/env python3
"""
Expectation maximization algorithm with GMM
"""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation-maximization (EM) algorithm for a Gaussian Mixture
    Model (GMM) on a given dataset.

    Parameters:
    - X (numpy.ndarray): dataset to be clustered, of shape (n, d) where n is
    the number of data points and d is the dimensionality of each data point.
    - k (int): The number of clusters.
    - iterations (int, optional): The maximum number of iterations for the
    algorithm (default is 1000).
    - tol (float, optional): The tolerance of the log likelihood for early
    stopping (default is 1e-5).
    - verbose (bool, optional): If True, prints the log likelihood after every
    10 iterations and after the last iteration (default is False).

    Returns:
    - pi (numpy.ndarray): The priors for each cluster, of shape (k,).
    - m (numpy.ndarray): The centroid means for each cluster, of shape (k, d).
    - S (numpy.ndarray): The covariance matrices for each cluster, of
    shape (k, d, d).
    - g (numpy.ndarray): The posterior probabilities for each data point in
    each cluster, of shape (k, n).
    - l (float): The log likelihood of the model.

    If the function fails, it returns (None, None, None, None, None).
    """
    if (
        not isinstance(X, np.ndarray) or X.ndim != 2
        or not isinstance(k, int) or k <= 0
        or not isinstance(iterations, int) or iterations <= 0
        or not isinstance(tol, float) or tol < 0
        or not isinstance(verbose, bool)
    ):
        return None, None, None, None, None

    # Initialize priors, centroid means, and covariance matrices
    pi, m, S = initialize(X, k)

    for i in range(iterations):
        # Evaluate the probabilities and likelihoods with current parameters
        g, prev_li = expectation(X, pi, m, S)

        # In verbose mode, print the likelihood every 10 iterations after 0
        if verbose and i % 10 == 0:
            print(f"Log Likelihood after {i} iterations: {round(prev_li, 5)}")

        # Re-estimate the parameters with the new values
        pi, m, S = maximization(X, g)

        # Evaluate new log likelihood
        g, li = expectation(X, pi, m, S)

        # If the likelihood varied by less than the tolerance value, we stop
        if np.abs(li - prev_li) <= tol:
            break

    # Last verbose message with current likelihood
    if verbose:
        # NOTE i + 1 since it has been updated once more since last print
        print(f"Log Likelihood after {i + 1} iterations: {round(li, 5)}")
    return pi, m, S, g, li
