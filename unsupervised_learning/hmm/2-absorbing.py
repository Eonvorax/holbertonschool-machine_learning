#!/usr/bin/env python3
"""
Absorbing Markov Chain
"""

import numpy as np


def absorbing(P):
    """
    Determines if a Markov chain is absorbing.

    Parameters:
    - P: numpy.ndarray of shape (n, n), representing the transition matrix.

    Returns:
    - True if the Markov chain is absorbing, False otherwise.
    """
    if not isinstance(P, np.ndarray) or P.shape[0] != P.shape[1]:
        return None
    if P.ndim != 2:
        return None
    if not np.allclose(P.sum(axis=1), 1):
        return None

    absorbing_states = np.where(np.isclose(P.diagonal(), 1))[0]
    if len(absorbing_states) == 0:
        # No absorbing states found
        return False

    # Non-absorbing states: NOTE see setdiff1d docs, finds unique values
    non_absorbing_states = np.setdiff1d(
        np.arange(P.shape[0]), absorbing_states)

    # Extracting submatrix Q, for non-absorbing states
    Q = P[(non_absorbing_states[:, None], non_absorbing_states)]

    # Identity matrix, the same size as Q
    Id = np.eye(Q.shape[0])

    try:
        # Calculate fundamental matrix
        N = np.linalg.inv(Id - Q)

        # If no elements are negative, it's an absorbing chain
        return np.all(N >= 0)
    except np.linalg.LinAlgError:
        # (I - Q) is not invertible
        return False
