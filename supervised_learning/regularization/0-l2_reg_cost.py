#!/usr/bin/env python3
"""
L2 regularization.
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.

    Parameters:
    - cost: original cost of the network without L2 regularization (float).
    - lambtha: The regularization parameter (float).
    - weights: dictionary of weights and biases of the NN (numpy.ndarrays).
    - L: The number of layers in the neural network (int).
    - m: The number of data points used (int).

    Returns:
    - The cost of the network accounting for L2 regularization (float).
    """
    l2_reg_term = 0

    for i in range(1, L + 1):
        # Accessing weights dictionary at key W{i}
        W = weights[f"W{str(i)}"]
        # Squared l2 norm: sum of squares of W values
        l2_reg_term += np.sum(np.square(W))
        # NOTE Same thing:
        # l2_reg_term += np.linalg.norm(W) ** 2

    l2_reg_term *= (lambtha / (2 * m))

    return cost + l2_reg_term
