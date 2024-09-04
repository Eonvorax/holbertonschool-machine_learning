#!/usr/bin/env python3
"""
Deep RNN
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN.

    :Parameters:
    - rnn_cells (list): A list of instances of `RNNCell` used for
    forward propagation (one for each layer)
    - X (numpy.ndarray): Data input for the RNN of shape (t, m, i).
        - t is the maximum number of time steps.
        - m is the batch size.
        - i is the dimensionality of the data.
    - h_0 (numpy.ndarray): Initial hidden state of shape (l, m, h).
        - l is the number of layers.
        - m is the batch size.
        - h is the dimensionality of the hidden state.

    :Returns:
    - H (numpy.ndarray): Hidden states for each layer for each time step,
    of shape (t+1, l, m, h). Includes the initial hidden states.
    - Y (numpy.ndarray): Outputs for each time step from the last layer,
    shape (t, m, o).
    """
    t, m, _ = X.shape
    l, _, h = h_0.shape

    # NOTE t + 1 time steps (counting initial hidden state), t outputs
    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, rnn_cells[-1].Wy.shape[1]))

    # Store initial hidden state
    H[0] = h_0

    for step in range(t):
        for layer, cell in enumerate(rnn_cells):
            # NOTE don't really need y except at layer == l, could be improved?
            # After experimenting, it would be better to modify forward()
            # ...which we're not allowed to do here.
            if layer == 0:
                # First layer, input data at current time step
                h_next, y = cell.forward(H[step, layer], X[step])
            else:
                # Following layers use hidden state from previous layer
                h_next, y = cell.forward(H[step, layer], h_next)
            # Store next hidden state
            H[step + 1, layer] = h_next
        # Store output of last layer
        Y[step] = y

    return H, Y
