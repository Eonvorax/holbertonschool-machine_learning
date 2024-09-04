#!/usr/bin/env python3
"""
Bidirectional Cell, forward direction
"""
import numpy as np


class BidirectionalCell:
    """
    This class represents a bidirectional cell of a RNN
    """
    def __init__(self, i, h, o):
        """
        Initialize a bidirectional cell for an RNN.

        :Parameters:
        - i (int): Dimensionality of the data.
        - h (int): Dimensionality of the hidden states.
        - o (int): Dimensionality of the outputs.

        :Attributes:
        - Whf (numpy.ndarray): Weight matrix for forward hidden state.
        - Whb (numpy.ndarray): Weight matrix for backward hidden state.
        - Wy (numpy.ndarray): Weight matrix for output.
        - bhf (numpy.ndarray): Biases for forward hidden state.
        - bhb (numpy.ndarray): Biases for backward hidden state.
        - by (numpy.ndarray): Biases for output.
        """
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h * 2, o)  # NOTE twice h: forward & backward
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for one time step.

        :Parameters:
        - h_prev (numpy.ndarray): Previous hidden state of shape (m, h).
        - x_t (numpy.ndarray): Input data of shape (m, i).

        :Returns:
        - numpy.ndarray: Next hidden state of shape (m, h * 2).
        """
        # Concatenate previous hidden state and cell data input
        concat_h_x = np.concatenate((h_prev, x_t), axis=1)

        # Return next (forward) hidden state
        return np.tanh(np.dot(concat_h_x, self.Whf) + self.bhf)
