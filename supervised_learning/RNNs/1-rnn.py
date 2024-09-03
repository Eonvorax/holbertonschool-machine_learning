#!/usr/bin/env python3
"""
Simple RNN forward propagation
"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN.

    :Parameters:
    - rnn_cell (RNNCell): An instance of `RNNCell` used for forward propagation
    - X (numpy.ndarray): Data input for the RNN of shape (t, m, i).
        - t is the maximum number of time steps.
        - m is the batch size.
        - i is the dimensionality of the data.
    - h_0 (numpy.ndarray): Initial hidden state of shape (m, h).
        - m is the batch size.
        - h is the dimensionality of the hidden state.

    :Returns:
    - H (numpy.ndarray): Hidden states for each time step of shape (t+1, m, h).
    Includes the initial hidden state.
    - Y (numpy.ndarray): Outputs for each time step of shape (t, m, o).
    """
    t, m, _ = X.shape
    _, h = h_0.shape

    # NOTE t + 1 time steps (counting initial hidden state), t outputs
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, rnn_cell.Wy.shape[1]))

    # Store initial hidden state
    H[0] = h_0

    for step in range(t):
        # Forward propagation using our rnn_cell
        h_next, y = rnn_cell.forward(H[step], X[step])

        # Store next hidden state
        H[step + 1] = h_next

        # Store output
        Y[step] = y

    return H, Y
