#!/usr/bin/env python3
"""
Calculate sensitivity
"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): confusion matrix, shape (classes, classes)
        where row indices represent the correct labels and column indices
        represent the predicted labels.

    Returns:
        numpy.ndarray: An array of shape (classes,) containing the sensitivity
        of each class.
    """
    # Number of classes
    classes = confusion.shape[0]
    # initialize sensitivity array at 0
    sensitivities = np.zeros((classes,))

    for i in range(classes):
        # True positives for class i
        true_positives = confusion[i, i]
        # Sum elements in i-th row
        all_positives = np.sum(confusion[i, :])
        # Sensitivity: ratio of TP over P
        sensitivities[i] = true_positives / all_positives

    return sensitivities
