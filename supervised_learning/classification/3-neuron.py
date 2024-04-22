#!/usr/bin/env python3

"""
This is the 2-neuron module.
"""
import numpy as np


class Neuron():
    """
    The Neuron class represents a single neuron in a neural network.

    Attributes:
        nx (int): The number of input features for the neuron.
        W (numpy.ndarray): The weights vector for the neuron.
        b (float): The bias for the neuron.
        A (float): The activated output of the neuron (prediction).

    Methods:
        __init__(self, nx): Initializes a Neuron instance with a given
        number of input features.
        W(self): Getter method for the weights vector.
        b(self): Getter method for the bias.
        A(self): Getter method for the activated output.
    """

    def __init__(self, nx):
        """
        Neuron class constructor.

        Args:
            nx (int): The number of input features for the neuron.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is not a positive integer.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        Getter method for the weights vector.

        Returns:
            numpy.ndarray: The weights vector for the neuron.
        """
        return self.__W

    @property
    def b(self):
        """
        Getter method for the bias.

        Returns:
            float: The bias for the neuron.
        """
        return self.__b

    @property
    def A(self):
        """
        Getter method for the activated output.

        Returns:
            float: The activated output of the neuron (prediction).
        """
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron.
        X is a numpy.ndarray with shape (nx, m) that contains the input data;
            nx is the number of input features to the neuron
            m is the number of examples
        """
        # Weighted sum + bias
        weighted_sum = np.matmul(self.W, X) + self.b

        # Sigmoid activation function
        self.__A = 1.0 / (1.0 + np.exp(-weighted_sum))

        return self.A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.
        Y is a numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data.
        A is a numpy.ndarray with shape (1, m) containing the activated
            output of the neuron for each example.
        """
        # to avoid division by 0 errors, using 1.0000001 - A instead of 1 - A
        return -np.mean(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
