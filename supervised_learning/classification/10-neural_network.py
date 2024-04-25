#!/usr/bin/env python3

"""
This is the NeuralNetwork class module.
"""
import numpy as np


class NeuralNetwork:
    """
    Represents a neural network with one hidden layer performing binary
    classification.

    Attributes:
        W1 (numpy.ndarray): The weights vector for the hidden layer.
        b1 (numpy.ndarray): The bias for the hidden layer.
        A1 (float): The activated output for the hidden layer.
        W2 (numpy.ndarray): The weights vector for the output neuron.
        b2 (float): The bias for the output neuron.
        A2 (float): The activated output for the output neuron (prediction).
    """

    def __init__(self, nx, nodes):
        """
        Initializes a neural network with one hidden layer performing
        binary classification.

        Args:
            nx (int): Number of input features.
            nodes (int): Number of nodes in the hidden layer.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter for W1."""
        return self.__W1

    @property
    def b1(self):
        """Getter for b1."""
        return self.__b1

    @property
    def A1(self):
        """Getter for A1."""
        return self.__A1

    @property
    def W2(self):
        """Getter for W2."""
        return self.__W2

    @property
    def b2(self):
        """Getter for b2."""
        return self.__b2

    @property
    def A2(self):
        """Getter for A2."""
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.

        Args:
        X (numpy.ndarray): Input data with shape (nx, m), where:
            - nx is the number of input features.
            - m is the number of examples.

        Returns:
        tuple: A tuple containing:
            - A1 (numpy.ndarray): The activated output for the hidden layer.
            - A2 (numpy.ndarray): The activated output for the output neuron.
        """
        # First layer (hidden layer) using the input data X
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        # Second layer (output layer) using A1 activation as input
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2
