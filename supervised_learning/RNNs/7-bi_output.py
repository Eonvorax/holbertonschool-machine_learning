#!/usr/bin/env python3
"""
Bidirectional Output
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
        - numpy.ndarray: the next hidden state
        """
        # Concatenate previous hidden state and cell data input
        concat_h_x = np.concatenate((h_prev, x_t), axis=1)

        # Return next (forward) hidden state
        return np.tanh(np.dot(concat_h_x, self.Whf) + self.bhf)

    def backward(self, h_next, x_t):
        """
        Perform backward propagation for one time step.

        :Parameters:
        - h_next (numpy.ndarray): Next hidden state of shape (m, h).
        - x_t (numpy.ndarray): Input data of shape (m, i).

        :Returns:
        - numpy.ndarray: the previous hidden state
        """
        # Concatenate next hidden state and cell data input
        concat_h_x = np.concatenate((h_next, x_t), axis=1)

        # Return previous (backward) hidden state
        return np.tanh(np.dot(concat_h_x, self.Whb) + self.bhb)

    @staticmethod
    def softmax(x):
        """
        Simple softmax method
        """
        return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)

    def output(self, H):
        """
        Calculates all outputs for the bidirectional RNN.

        :Parameters:
        - H (numpy.ndarray) of shape `(t, m, 2 * h)` that contains the
        concatenated hidden states from both directions, excluding their
        initialized states.
            - t is the number of time steps
            - m is the batch size for the data
            - h is the dimensionality of the hidden states

        :Returns:
        - Y (numpy.ndarray) the outputs
        """
        t, m, _ = H.shape
        Y = np.zeros((t, m, self.Wy.shape[1]))

        for step in range(t):
            # Softmax-activated output using each hidden state
            Y[step] = self.softmax(H[step] @ self.Wy + self.by)

        return Y
