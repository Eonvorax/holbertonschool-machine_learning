#!/usr/bin/env python3
"""
Markov Chain
"""

import numpy as np


def markov_chain(P, s, t=1):
    """
    Determines the probability of a Markov chain being in a particular
    state after a specified number of iterations.

    Parameters:
    - P: numpy.ndarray of shape `(n, n)`, representing the transition matrix.
    - s: numpy.ndarray of shape `(1, n)`, representing the initial state
    probability distribution.
    - t: int, number of iterations the Markov chain has been through.

    Returns:
    - numpy.ndarray of shape `(1, n)` representing the probability distribution
    after t iterations, or `None` if the inputs are invalid.
    """
    if not isinstance(P, np.ndarray) or not isinstance(s, np.ndarray):
        return None
    if P.ndim != s.ndim:
        return None
    if s.shape[1] != P.shape[0] or s.shape[1] != P.shape[1]:
        return None
    if not isinstance(t, int) or t < 0:
        return None
    if not np.allclose(P.sum(axis=1), 1) or not np.isclose(s.sum(), 1):
        return None

    # k-th state matrix: S{k} = S{0}.P^k, with k=t in this case
    return np.dot(s, np.linalg.matrix_power(P, t))
