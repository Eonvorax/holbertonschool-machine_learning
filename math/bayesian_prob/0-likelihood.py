#!/usr/bin/env python3

"""
Likelyhood
"""
import numpy as np


def likelihood(x, n, P):
    """
    Calculate the likelihood of obtaining the observed data given various
    hypothetical probabilities of developing severe side effects.

    Parameters:
    - x (int): The number of patients that develop severe side effects.
    - n (int): The total number of patients observed.
    - P (numpy.ndarray): A 1D numpy.ndarray containing the various
    hypothetical probabilities of developing severe side effects.

    Returns:
    - numpy.ndarray: A 1D numpy.ndarray containing the likelihood of obtaining
    the data, x and n, for each probability in P, respectively.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    # Logical "or" using numpy vectorized operation
    if any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    return binomial_coefficient(n, x) * (P ** x) * ((1 - P) ** (n - x))


def binomial_coefficient(n, k):
    """Compute the binomial coefficient using numpy factorial."""
    return np.math.factorial(n) /\
        (np.math.factorial(k) * np.math.factorial(n - k))
