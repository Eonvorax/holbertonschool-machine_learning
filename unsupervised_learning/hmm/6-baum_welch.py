#!/usr/bin/env python3
"""
The Baum-Welch Algorithm (Expectation-Maximization)
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


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for a hidden Markov model.
    """
    if (not isinstance(Observations, np.ndarray) or Observations.ndim != 1 or
            not isinstance(Emission, np.ndarray) or Emission.ndim != 2 or
            not isinstance(Transition, np.ndarray) or Transition.ndim != 2 or
            not isinstance(Initial, np.ndarray) or Initial.ndim != 2):
        return None, None

    N = Transition.shape[0]
    M = Emission.shape[1]
    T = Observations.shape[0]

    for _ in range(iterations):
        # Forward and Backward passes
        P_f, F = forward(Observations, Emission, Transition, Initial)
        P_b, B = backward(Observations, Emission, Transition, Initial)

        # Initialize variables
        xi = np.zeros((N, N, T-1))
        gamma = np.zeros((N, T))

        for t in range(T-1):
            # Broadcast computation across all states
            # NOTE newaxis to match column vectors in calculation
            xi[:, :, t] = (F[:, t, np.newaxis] * Transition *
                           Emission[:, Observations[t+1]] * B[:, t+1]) / P_f

        gamma = np.sum(xi, axis=1)

        # Need final gamma element for new B
        prod = (F[:, T-1] * B[:, T-1]).reshape((-1, 1))
        gamma = np.hstack((gamma,  prod / np.sum(prod)))

        # Re-estimate Transition matrix
        Transition = np.sum(xi, axis=2) / \
            np.sum(gamma[:, :-1], axis=1).reshape((-1, 1))

        # Re-estimate Emission matrix
        for k in range(M):
            Emission[:, k] = np.sum(gamma[:, Observations == k], axis=1)

        Emission /= np.sum(gamma, axis=1).reshape(-1, 1)

    return Transition, Emission
