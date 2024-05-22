#!/usr/bin/env python3
"""
Gradient Descent with Dropout.
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a NN with Dropout regularization
    using gradient descent.

    Parameters:
    Y (numpy.ndarray): One-hot array of shape (classes, m) with correct
        labels for the data
    weights (dict): Dictionary of the weights and biases of the NN
    cache (dict): Dictionary of the outputs and dropout masks of each
        layer of the NN
    alpha (float): Learning rate
    keep_prob (float): Probability that a node will be kept
    L (int): Number of layers of the network

    Returns:
    None: Updates the weights dictionary in place
    """
    m = Y.shape[1]
    dZ = cache[f"A{L}"] - Y

    # In reverse layer order :
    for i in range(L, 0, -1):
        # Previous layer activation output
        prev_A = cache[f"A{i - 1}"]

        # Gradient of loss with respect to weights and biases
        dW = (1 / m) * np.matmul(dZ, prev_A.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            # Gradient of the activation function (tanh)
            dA = np.matmul(weights[f"W{i}"].T, dZ)
            # Apply dropout mask from cache
            dA *= cache[f"D{i - 1}"]
            # Scale back activation gradients by keep probability
            dA /= keep_prob
            # Apply tanh gradient
            dZ = dA * (1 - np.square(cache[f"A{i - 1}"]))

        # Updating weights and biases
        weights[f"W{i}"] -= alpha * dW
        weights[f"b{i}"] -= alpha * db
