#!/usr/bin/env python3

"""
This is the DeepNeuralNetwork class module.
"""
import numpy as np


class DeepNeuralNetwork:
    """
    Represents a deep neural network to perform classification.
    """

    def __init__(self, nx, layers):
        """
        Initialize the deep neural network object.

        Args:
            nx (int): The number of input features.
            layers (list): A list of positive integers representing the
                number of nodes in each layer.

        Raises:
            TypeError: If nx is not an integer or layers is not a list of
                positive integers.
            ValueError: If nx is not a positive integer.

        Attributes:
            L (int): The number of layers in the neural network.
            cache (dict): A dictionary to hold intermediate values.
            weights (dict): A dictionary to hold the weights and biases.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda x: isinstance(x, int) and x > 0, layers)):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if i == 0:
                # On layer 0: first layer's length is nx
                prev_layer = nx
            else:
                # Otherwise we use the previous layer length
                prev_layer = layers[i - 1]

            # He Normal (He-et-al) initialization
            self.__weights[f"W{i + 1}"] = np.random.randn(
                layers[i], prev_layer) * np.sqrt(2 / prev_layer)

            self.__weights[f"b{i + 1}"] = np.zeros((layers[i], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.
        """
        # Setting up the first input of first layer : X
        self.__cache["A0"] = X

        for i in range(1, self.__L + 1):
            # Previous layer activation ouput, used as input
            prev_A = self.__cache[f"A{i - 1}"]

            Z = np.matmul(self.__weights[f"W{i}"], prev_A)\
                + self.__weights[f"b{i}"]

            # Apply sigmoid and store activation in cache
            self.__cache[f"A{i}"] = 1 / (1 + np.exp(-Z))

        # The output of the network is in A{layer} (the last output)
        return self.__cache[f"A{self.__L}"], self.__cache
