#!/usr/bin/env python3
"""
Epsilon-greedy policy
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Determines the next action using the epsilon-greedy policy.

    Parameters:
        Q (numpy.ndarray): The Q-table, where each entry Q[s, a] represents
            the expected reward for state `s` and action `a`.
        state (int): The current state.
        epsilon (float): The epsilon value for the epsilon-greedy policy.
            With probability `epsilon` the action is chosen randomly
            (explore) and with probability `(1 - epsilon)` the action with the
            highest Q-value is chosen (exploit).

    Returns:
        int: The index of the action to take next.
    """
    # With a sampled random value between 0 and 1:
    if np.random.uniform(0, 1) > epsilon:
        # Exploit: choose the action with the highest Q-value
        return np.argmax(Q[state, :])
    else:
        # Explore: choose a random action
        return np.random.randint(Q.shape[1])
