#!/usr/bin/env python3
"""
Gated Recurrent Unit
"""

import numpy as np


class GRUCell:
    """
    This class represents the Gated Recurrent Unit of a RNN
    """

    def __init__(self, i, h, o):
        """
        Initialize the GRU.

        :Parameters:
        - i (int): Dimensionality of the data.
        - h (int): Dimensionality of the hidden state.
        - o (int): Dimensionality of the outputs.
        """
        # Weigths initialized with a random normal distribution
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        # biases are initialized with zeros
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
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

        # Update gate output
        z_t = self.sigmoid(np.dot(concat_h_x, self.Wz) + self.bz)

        # Reset gate output
        r_t = self.sigmoid(np.dot(concat_h_x, self.Wr) + self.br)

        # Candidate hidden state h^_t (using reset gate)
        concat_r_h_x = np.concatenate((r_t * h_prev, x_t), axis=1)
        h_candidate = np.tanh(np.dot(concat_r_h_x, self.Wh) + self.bh)

        # Next hidden state h_next
        h_next = (1 - z_t) * h_prev + z_t * h_candidate

        # Calculate output using softmax activation function
        y_linear = np.dot(h_next, self.Wy) + self.by
        y = self.softmax(y_linear)

        # Return both the hidden state and the softmax-activated output
        return h_next, y

    @staticmethod
    def softmax(x):
        """
        Simple softmax method
        """
        return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)

    @staticmethod
    def sigmoid(x):
        """
        Simple sigmoid method
        """
        return 1 / (1 + np.exp(-x))
