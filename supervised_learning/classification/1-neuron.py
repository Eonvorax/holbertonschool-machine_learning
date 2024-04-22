#!/usr/bin/env python3

"""
This is the 1-neuron module.
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

        self.__W = np.random.randn(nx).reshape(1, nx)
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
