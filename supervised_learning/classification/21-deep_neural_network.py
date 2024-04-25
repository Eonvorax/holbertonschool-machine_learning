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

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions.

        X is a numpy.ndarray with shape (nx, m) that contains the input data.
        nx is the number of input features to the neuron.
        m is the number of examples.
        Y is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data.

        Returns the NN's prediction (a numpy.ndarray with shape (1, m))
        and the cost of the network, respectively.
        """
        self.forward_prop(X)

        # If elt of A >= 0.5 then 1, otherwise 0 (same threshold of 0.5)
        prediction = np.where(self.__cache[f"A{self.__L}"] >= 0.5, 1, 0)

        # Cost of activated output on "output" layer
        cost = self.cost(Y, self.__cache[f"A{self.__L}"])

        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the network.

        Y is a numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data
        cache is a dictionary containing all the intermediary values
        alpha is the learning rate

        Updates the private attribute __weights
        """

        m = Y.shape[1]
        dZ = cache[f"A{self.__L}"] - Y

        # In reverse layer order :
        for i in range(self.__L, 0, -1):
            # Previous layer activation output
            prev_A = cache[f"A{i - 1}"]

            dW = np.matmul(dZ, prev_A.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            if i > 1:
                # Prepare the next layer's gradient calculation
                dZ = np.matmul(
                    self.__weights[f"W{i}"].T, dZ) * prev_A * (1 - prev_A)

            # Updating parameters using the gradients and the learning rate
            self.__weights[f"W{i}"] -= alpha * dW
            self.__weights[f"b{i}"] -= alpha * db
