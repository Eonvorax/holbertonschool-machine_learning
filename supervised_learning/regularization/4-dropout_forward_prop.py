#!/usr/bin/env python3
"""
Forward Propagation with Dropout.
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.

    Parameters:
    X (numpy.ndarray): Input data for the network of shape (nx, m)
        nx is the number of input features
        m is the number of data points
    weights (dict): Dictionary of the weights and biases of the neural network
    L (int): Number of layers in the network
    keep_prob (float): Probability that a node will be kept

    Returns:
    dict: Outputs of each layer and the dropout mask used on each layer
    """
    # Setting up the first input of first layer : X
    cache = {'A0': X}

    for i in range(1, L + 1):
        # Previous layer activation ouput, used as input
        prev_A = cache[f"A{i - 1}"]
        Z = np.matmul(weights[f"W{i}"], prev_A) + weights[f"b{i}"]

        # Apply tanh activation and dropout to all layers except the last one
        if i < L:
            A = np.tanh(Z)
            # Create dropout mask and add it to the cache
            D = np.random.binomial(1, keep_prob, size=A.shape)
            cache[f"D{i}"] = D
            # Apply dropout mask and scale A back from 0.8
            A = np.multiply(A, D)
            A /= keep_prob
        else:
            # For the output layer, use softmax activation (no dropout)
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

        # Store activation in cache
        cache[f"A{i}"] = A

    # Returning the resulting cache
    return cache
