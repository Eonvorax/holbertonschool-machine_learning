#!/usr/bin/env python3
"""
Gradient descent with L2 regularization.
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient descent
    with L2 regularization.

    Parameters:
    - Y: A one-hot numpy.ndarray of shape (classes, m) that contains the
        correct labels for the data
    - weights: A dictionary of the weights and biases of the neural network
    - cache: A dictionary of the outputs of each layer of the neural network
    - alpha: The learning rate (float)
    - lambtha: The L2 regularization parameter (float)
    - L: The number of layers of the network (int)

    The neural network uses tanh activations on each layer except the last,
    which uses a softmax activation.
    The weights and biases of the network are updated in place.
    """
    m = Y.shape[1]
    dZ = cache[f"A{L}"] - Y

    # In reverse layer order :
    for i in range(L, 0, -1):
        # Previous layer activation output
        prev_A = cache[f"A{i - 1}"]

        # L2 regularization term
        l2_reg = (lambtha / m) * weights[f"W{i}"]

        # Gradient of loss with respect to weights and biases
        dW = (1 / m) * np.matmul(dZ, prev_A.T) + l2_reg
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            # Gradient of the activation function (tanh)
            dZ = np.matmul(weights[f"W{i}"].T, dZ) * (1 - np.square(prev_A))

        # Updating weights and biases
        weights[f"W{i}"] -= alpha * dW
        weights[f"b{i}"] -= alpha * db
