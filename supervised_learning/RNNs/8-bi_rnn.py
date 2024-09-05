#!/usr/bin/env python3
"""
Bidirectional RNN
"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN.

    Parameters:
    - bi_cell (BidirectionalCell): Instance of BidirectionalCell used for
    forward propagation.
    - X (numpy.ndarray): Data input of shape (t, m, i).
        - t is the number of time steps.
        - m is the batch size.
        - i is the dimensionality of the data.
    - h_0 (numpy.ndarray): Initial hidden state in the forward direction,
    of shape (m, h).
    - h_t (numpy.ndarray): Initial hidden state in the backward direction,
    of shape (m, h).

    Returns:
    - H (numpy.ndarray): Concatenated hidden states for each time step, of
    shape (t, m, 2 * h).
    - Y (numpy.ndarray): Outputs for each time step, of shape (t, m, o).
    """
    t, m, _ = X.shape
    _, h = h_0.shape

    # NOTE twice h: forward & backward
    H = np.zeros((t, m, h * 2))
    Y = np.zeros((t, m, bi_cell.Wy.shape[1]))

    # Initial forward and backward hidden states
    h_f = h_0
    h_b = h_t

    for step in range(t):
        # Forward pass
        h_f = bi_cell.forward(h_f, X[step])
        # Store forward hidden state
        H[step, :, :h] = h_f

        # Backward pass (starting from the end)
        h_b = bi_cell.backward(h_b, X[-1 - step])
        # Store backward hidden state
        H[-1 - step, :, h:] = h_b

    Y = bi_cell.output(H)

    return H, Y
