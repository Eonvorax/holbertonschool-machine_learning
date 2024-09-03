#!/usr/bin/env python3
"""
Long Short-Term Memory unit
"""

import numpy as np


class LSTMCell:
    """
    This class represents the Long Short-Term Memory unit of a RNN
    """

    def __init__(self, i, h, o):
        """
        Initialize the LSTM unit.

        :Parameters:
        - i (int): Dimensionality of the data.
        - h (int): Dimensionality of the hidden state.
        - o (int): Dimensionality of the outputs.

        :Public attributes (weights and biases):
        - Wf and bf are for the forget gate
        - Wu and bu are for the update gate
        - Wc and bc are for the intermediate cell state
        - Wo and bo are for the output gate
        - Wy and by are for the outputs
        """
        # Weigths initialized with a random normal distribution
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        # biases are initialized with zeros
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Perform forward propagation for one time step.

        :Parameters:
        - h_prev (numpy.ndarray): Previous hidden state of shape (m, h).
        - c_prev (numpy.ndarray): Previous cell state of shape (m, h).
        - x_t (numpy.ndarray): Data input for the cell of shape (m, i).

        :Returns:
        - h_next (numpy.ndarray): Next hidden state.
        - c_next (numpy.ndarray): Next cell state.
        - y (numpy.ndarray): softmax-activated output of the cell.
        """
        # Concatenate previous hidden state and cell data input
        concat_h_x = np.concatenate((h_prev, x_t), axis=1)

        # Forget gate
        f_t = self.sigmoid(np.dot(concat_h_x, self.Wf) + self.bf)

        # Update gate
        u_t = self.sigmoid(np.dot(concat_h_x, self.Wu) + self.bu)

        # Output gate
        o_t = self.sigmoid(np.dot(concat_h_x, self.Wo) + self.bo)

        # New cell input activation vector
        c_act = np.tanh(np.dot(concat_h_x, self.Wc) + self.bc)

        # New cell state
        c_next = f_t * c_prev + u_t * c_act

        # hidden state
        h_next = o_t * np.tanh(c_next)

        # Calculate output using softmax activation function
        y_linear = np.dot(h_next, self.Wy) + self.by
        y = self.softmax(y_linear)

        # Return next hidden state, cell state and softmax-activated output
        return h_next, c_next, y

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
