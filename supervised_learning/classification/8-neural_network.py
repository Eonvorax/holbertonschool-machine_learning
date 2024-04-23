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

        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0

        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
