#!/usr/bin/env python3
"""
Calculates the weighted moving average of a dataset.
"""

import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Update a variable using the gradient descent with momentum
    optimization algorithm.

    Args:
        alpha (float): The learning rate.
        beta1 (float): The momentum weight.
        var (numpy.ndarray): The variable to be updated.
        grad (numpy.ndarray): The gradient of the variable.
        v (numpy.ndarray): The previous first moment of the variable.

    Returns:
        tuple: The updated variable and the new moment, respectively.
    """
    # Compute new momentum
    updated_v = beta1 * v + (1 - beta1) * grad

    # Compute the new variable using new momentum and current var
    updated_var = var - alpha * updated_v

    return updated_var, updated_v
