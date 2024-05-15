#!/usr/bin/env python3
"""
Computes RMSProp optimization algorithm.
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Update a variable using the RMSProp optimization algorithm.

    Args:
        alpha (float): The learning rate.
        beta2 (float): The RMSProp weight.
        epsilon (float): A small number to avoid division by zero.
        var (numpy.ndarray): The variable to be updated.
        grad (numpy.ndarray): The gradient of var.
        s (numpy.ndarray): The previous second moment of var.

    Returns:
        tuple: The updated variable and the new moment, respectively.
    """
    # Compute new moment
    updated_s = beta2 * s + (1 - beta2) * np.square(grad)

    # Compute the new variable using new momentum and current var
    updated_var = var - alpha * grad / (np.sqrt(updated_s) + epsilon)

    return updated_var, updated_s
