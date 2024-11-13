#!/usr/bin/env python3
"""
Compute the Monte-Carlo policy gradient
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


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient based on a state and a
    weight matrix.

    Parameters:
        state: matrix representing the current observation of the
            environment
        weight: matrix of random weight

    Returns:
        tuple ((int, np.ndarray)): the chosen action, and the gradient
    """

    # Compute action probabilities using the policy function
    action_probs = policy(state, weight)

    # Sample an action based on these probabilities
    action = np.random.choice(len(action_probs), p=action_probs)

    d_softmax = action_probs.copy()

    # NOTE one-hot encoding adjustment on the action probability
    # to reflect a gradient where the chosen action is favored
    # gradient_vector = one-hot(a) − action_probs
    d_softmax[action] -= 1

    # Gradient with respect to weights:
    # negative outer product of state and adjusted probabilities
    grad = -np.outer(state, d_softmax)

    return action, grad
