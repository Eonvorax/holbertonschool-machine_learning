#!/usr/bin/env python3
"""
The Viterbi Algorithm
"""

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculates the most likely sequence of hidden states for a hidden Markov
    model, using the Viterbi Algorithm.

    Parameters:
    - Observation: numpy.ndarray of shape `(T,)` that contains the index of the
    observation
    - Emission: numpy.ndarray of shape `(N, M)` containing the emission
    probability of a specific observation given a hidden state
    - Transition: 2D numpy.ndarray of shape `(N, N)` containing the transition
    probabilities
    - Initial: numpy.ndarray of shape `(N, 1)` containing the probability of
    starting in a particular hidden state

    Returns:
    - path: list of length `T` containing the most likely sequence of hidden
    states
    - P: the probability of obtaining the path sequence
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

    # Initialize the Viterbi and state tracking matrices
    V = np.zeros((N, T))
    B = np.zeros((N, T), dtype=int)

    # Init step: fill the 1st column of V using Initial and 1st Observation
    V[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    # Recursion for following probs.
    for t in range(1, T):
        # Compute maximum probability & corresponding states for time t
        probabilities = V[:, t-1].reshape(-1, 1) * \
            Transition * Emission[:, Observation[t]]
        V[:, t] = np.max(probabilities, axis=0)
        # Tracking most likely state for future backtracking
        B[:, t] = np.argmax(probabilities, axis=0)

    # Finding the probability of the most likely path
    P = np.max(V[:, T-1])

    # Path backtracking from most likely path (starting from last index)
    last_state = np.argmax(V[:, T-1])
    path = [last_state]
    for t in range(T-1, 0, -1):
        last_state = B[last_state, t]
        path.append(last_state)
    path.reverse()

    return path, P
