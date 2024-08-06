#!/usr/bin/env python3
"""
The Forward Algorithm
"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden Markov model.

    Parameters:
    - Observation: numpy.ndarray of shape (T,) that contains the index of the
    observation
    - Emission: numpy.ndarray of shape (N, M) containing the emission
    probability of a specific observation given a hidden state
    - Transition: 2D numpy.ndarray of shape (N, N) containing the transition
    probabilities
    - Initial: numpy.ndarray of shape (N, 1) containing the probability of
    starting in a particular hidden state

    Returns:
    - P: likelihood of the observations given the model
    - F: numpy.ndarray of shape (N, T) containing the forward path
    probabilities
    """
    if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None

    N, M = Emission.shape
    T = Observation.shape[0]

    if Transition.shape != (N, N):
        return None, None
    if Initial.shape != (N, 1):
        return None, None

    # Forward probability matrix F, initialized with zeros
    F = np.zeros((N, T))

    # First column of F, filled using Initial and first Observation
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    # Recursion algorithm for the forward probabilities
    for t in range(1, T):
        # Use previous column of F to calc. next column
        F[:, t] = F[:, t-1] @ Transition * Emission[:, Observation[t]]

    # prob. of observation sequence is the sum of the last column in F
    P = np.sum(F[:, -1])

    return P, F
