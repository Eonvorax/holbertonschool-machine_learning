#!/usr/bin/env python3
"""
Early Stopping
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if gradient descent should be stopped early.

    Parameters:
    - cost: current validation cost of the neural network
    - opt_cost: lowest recorded validation cost of the neural network
    - threshold: threshold used for early stopping
    - patience: patience count used for early stopping
    - count: count of how long the threshold has not been met

    Returns: a boolean of whether the network should be stopped early,
        followed by the updated count
    """

    if (opt_cost - cost) <= threshold:
        count += 1
    else:
        count = 0

    return count >= patience, count
