#!/usr/bin/env python3

"""
This is the DeepNeuralNetwork class module.
"""
import numpy as np
import matplotlib.pyplot as plt


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
        if not np.all(map(lambda x: isinstance(x, int) and x > 0, layers)):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(self.L):
            if i == 0:
                # On layer 0: first layer's length is nx
                prev_layer = nx
            else:
                # Otherwise we use the previous layer length
                prev_layer = layers[i - 1]

            # He Normal (He-et-al) initialization
            self.weights[f"W{i + 1}"] = np.random.randn(layers[i], prev_layer)\
                * np.sqrt(2 / prev_layer)

            self.weights[f"b{i + 1}"] = np.zeros((layers[i], 1))
