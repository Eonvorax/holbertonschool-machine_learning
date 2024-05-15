#!/usr/bin/env python3
"""
Normalize an unactivated output (Z) of a NN using batch normalization.
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network using batch
    normalization.

    Args:
        Z (numpy.ndarray): The input matrix to be normalized, of shape (m, n).
        gamma (numpy.ndarray): The scale parameter of shape (1, n).
        beta (numpy.ndarray): The offset parameter of shape (1, n).
        epsilon (float): A small number used to avoid division by zero.

    Returns:
        numpy.ndarray: The normalized Z matrix.
    """
    # NOTE Similar to tasks 0 & 1
    mean = np.mean(Z, axis=0, keepdims=True)
    variance = np.var(Z, axis=0, keepdims=True)

    # Normalize using small epsilon value as usual (for num. stability)
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)

    # Z Matrix scaled by gamma and offset by beta
    return gamma * Z_norm + beta
