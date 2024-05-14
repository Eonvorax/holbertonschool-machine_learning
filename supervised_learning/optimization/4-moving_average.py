#!/usr/bin/env python3
"""
Calculates the weighted moving average of a dataset.
"""


def moving_average(data, beta):
    """
    Calculate the weighted moving average of a data set.
    NOTE: The moving average calculation uses bias correction.

    Args:
        data (list): The list of data to calculate the moving average of.
        beta (float): The weight used for the moving average.

    Returns:
        list: A list containing the moving averages of data.
    """
    moving_averages = []
    moving_avg = 0

    for i, x in enumerate(data):
        # Update current moving average (weighted by beta)
        moving_avg = beta * moving_avg + (1 - beta) * x
        # Bias correction: dividing avg by correction factor
        # NOTE (i + 1) to adjust for the zero-indexed loop iteration
        moving_avg_corrected = moving_avg / (1 - beta ** (i + 1))
        moving_averages.append(moving_avg_corrected)

    return moving_averages
