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
        # Would return a tuple (A1, A2), no need to assign A1 or A2 anyway
        self.forward_prop(X)

        # If elt of A >= 0.5 then 1, otherwise 0 (same threshold of 0.5)
        prediction = np.where(self.__A2 >= 0.5, 1, 0)

        # Cost of activated output on "output" layer
        cost = self.cost(Y, self.__A2)

        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network.

        X is a numpy.ndarray with shape (nx, m) that contains the input data
            nx is the number of input features to the neuron
            m is the number of examples
        Y is a numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data
        A1 is the output of the hidden layer
        A2 is the predicted output
        alpha is the learning rate

        Updates the private attributes __W1, __b1, __W2, and __b2
        """
        # NOTE is Y better here ?
        m = Y.shape[1]

        dZ2 = A2 - Y
        # Computing gradients of W2 and b2 over Z2
        dW2 = (1/m) * np.matmul(dZ2, A1.T)  # X.T == np.transpose(X)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.matmul(self.__W2.T, dZ2) * A1 * (1 - A1)
        # Computing gradients of W1 and b1 over Z1
        dW1 = (1/m) * np.matmul(dZ1, X.T)  # X.T == np.transpose(X)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        # Updating parameters using the gradients and the learning rates
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2

        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1
