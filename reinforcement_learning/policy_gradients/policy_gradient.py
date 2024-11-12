#!/usr/bin/env python3
"""
Simple stochastic policy function
"""
import numpy as np


def policy(matrix, weight):
    """
    Computes a stochastic policy by taking a weighted combination of the
    state and weight matrices and applying a softmax function.
    Parameters:
        matrix (np.ndarray): represents the current observation of the
            environment (states)
        weight (np.ndarray): weight matrix to apply for each state

    Returns:
        np.ndarray: Probability matrix over possible actions.
    """
    # Compute the linear combination of state and weights
    weighted_states = (matrix @ weight)

    # Apply softmax to the result to get probabilities
    # NOTE Stabilized by subtracting max value from weighted_states
    e_x = np.exp(weighted_states - np.max(weighted_states))
    return e_x / np.sum(e_x)
