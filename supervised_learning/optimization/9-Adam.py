#!/usr/bin/env python3
"""
Updates a variable in place using Adam optimization.
"""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Update a variable in place using the Adam optimization algorithm.

    Args:
        alpha (float): The learning rate.
        beta1 (float): The weight used for the first moment.
        beta2 (float): The weight used for the second moment.
        epsilon (float): A small number to avoid division by zero.
        var (numpy.ndarray): The variable to be updated.
        grad (numpy.ndarray): The gradient of var.
        v (numpy.ndarray): The previous first moment of var.
        s (numpy.ndarray): The previous second moment of var.
        t (int): The time step used for bias correction.

    Returns:
        tuple: The updated variable, the new first moment, and the new
        second moment, respectively.
    """
    # Update (biased) 1st & 2nd moments
    updated_v = beta1 * v + (1 - beta1) * grad
    updated_s = beta2 * s + (1 - beta2) * np.square(grad)

    # Bias-corrected 1st & 2nd moments
    v_corrected = updated_v / (1 - beta1 ** t)
    s_corrected = updated_s / (1 - beta2 ** t)

    # Compute the updated variable
    updated_var = var - alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)

    return updated_var, updated_v, updated_s
