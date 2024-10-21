#!/usr/bin/env python3
"""
Initialize Q-table
"""

import numpy as np


def q_init(env):
    """
    Initializes the Q-table for a given FrozenLake environment.

    Parameters:
        env (gym.Env): The FrozenLake environment instance.

    Returns:
        numpy.ndarray:
            A Q-table initialized to zeros with shape `(S, A)` where S is
            the number of states and A is the number of actions.
    """
    # Get number of states in the environment
    n_states = env.observation_space.n

    # Get number of actions in the environment
    n_actions = env.action_space.n

    # Initialize the Q-table (with zeros)
    return np.zeros((n_states, n_actions))
