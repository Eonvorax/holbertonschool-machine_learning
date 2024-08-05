#!/usr/bin/env python3
"""
Regular Markov Chain
"""

import numpy as np


def regular(P):
    """
    Determines the steady state probabilities of a regular Markov chain.

    Parameters:
    - P: numpy.ndarray of shape (n, n), representing the transition matrix.

    Returns:
    - numpy.ndarray of shape (1, n) containing the steady state probabilities,
      or None if the Markov chain is not regular or the input is invalid.
    """
    if not isinstance(P, np.ndarray) or P.shape[0] != P.shape[1]:
        return None
    if P.ndim != 2:
        return None
    if not np.allclose(P.sum(axis=1), 1):
        return None

    n = P.shape[0]
    # Checking if P is regular
    square_P = np.linalg.matrix_power(P, n**2)
    if not np.all((square_P > 0)):
        return None

    # NOTE steady-state dist. is the eigenvector associated w/ the eigenvalue 1
    # Calculate eigenvalues & eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(P.T)

    # Find the eigenvector corresponding to the eigenvalue 1
    index = np.argmin(np.abs(eigenvalues - 1))
    steady_state = eigenvectors[:, index]

    # Return the normalized value (a probability distribution)
    steady_state = steady_state / np.sum(steady_state)
    return steady_state.reshape(1, n)
