#!/usr/bin/env python3
"""
Calculate F1 score
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): confusion matrix, shape (classes, classes)
        where row indices represent the correct labels and column indices
        represent the predicted labels.

    Returns:
        numpy.ndarray: An array of shape (classes,) containing the F1 score
        of each class.
    """

    ppv = precision(confusion)
    tpr = sensitivity(confusion)
    # Weighted average of precision and sensitivity
    return (2 * ppv * tpr) / (ppv + tpr)
