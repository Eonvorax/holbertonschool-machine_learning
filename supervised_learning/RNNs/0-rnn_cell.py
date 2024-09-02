#!/usr/bin/env python3
"""
Basic RNN Cell
"""

import numpy as np


class RNNCell:
    """
    This class represents a cell of a simple RNN
    """

    def __init__(self, i, h, o):
        """
        Initialize the RNN cell.

        :Parameters:
        - i (int): Dimensionality of the data.
        - h (int): Dimensionality of the hidden state.
        - o (int): Dimensionality of the outputs.
        """
        # Weigths initialized with a random normal distribution
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        # biases are initialized with zeros
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for one time step.

        :Parameters:
        - h_prev (numpy.ndarray): Previous hidden state of shape (m, h).
        - x_t (numpy.ndarray): Data input for the cell of shape (m, i).

        :Returns:
        - h_next (numpy.ndarray): Next hidden state.
        - y (numpy.ndarray): softmax-activated output of the cell.
        """
        # Concatenate previous hiddent state and cell data input
        concat_h_x = np.concatenate((h_prev, x_t), axis=1)

        # Calculate next hidden state using tanh activation function
        h_next = np.tanh(np.dot(concat_h_x, self.Wh) + self.bh)

        # Calculate output using softmax activation function
        y_linear = np.dot(h_next, self.Wy) + self.by
        y = np.exp(y_linear) / np.exp(y_linear).sum(axis=1, keepdims=True)

        # Return both the hidden state and the softmax-activated output
        return h_next, y
