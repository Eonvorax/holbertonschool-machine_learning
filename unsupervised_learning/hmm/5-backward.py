#!/usr/bin/env python3
"""
Markov Chain Backward Algorithm
"""

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden markov model

    Parameters:
    - Observation: numpy.ndarray of shape `(T,)` containing the index of
    the observation
    - Emission: numpy.ndarray of shape `(N, M)` containing the emission
    probability of a specific observation given a hidden state
    - Transition: 2D numpy.ndarray of shape `(N, N)` containing the transition
    probabilities
    - Initial: numpy.ndarray of shape `(N, 1)` containing the probability of
    starting in a particular hidden state

    Returns:
    - P: the likelihood of the observations given the model
    - B: numpy.ndarray of shape (N, T) containing the backward path
    probabilities
    """
    if (not isinstance(Observation, np.ndarray) or Observation.ndim != 1 or
            not isinstance(Emission, np.ndarray) or Emission.ndim != 2 or
            not isinstance(Transition, np.ndarray) or Transition.ndim != 2 or
            not isinstance(Initial, np.ndarray) or Initial.ndim != 2):
        return None, None

    N, M = Emission.shape
    T = Observation.shape[0]

    if Transition.shape != (N, N):
        return None, None
    if Initial.shape != (N, 1):
        return None, None

    # Backward probability matrix B, initialized with zeros
    B = np.zeros((N, T))

    # Set the last column of B to 1 (initial state assumed as given)
    B[:, T - 1] = 1

    # Recursion: Fill B from time T-2 down to time 0
    for t in range(T - 2, -1, -1):
        # Calculate backward probabilities for states at time t
        B[:, t] = np.sum(
            Transition * Emission[:, Observation[t + 1]] * B[:, t + 1], axis=1)

    # Calculate likelihood of the observations, given the model
    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])

    return P, B
