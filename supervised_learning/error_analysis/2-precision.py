#!/usr/bin/env python3
"""
Calculate precision
"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): confusion matrix, shape (classes, classes)
        where row indices represent the correct labels and column indices
        represent the predicted labels.

    Returns:
        numpy.ndarray: An array of shape (classes,) containing the precision
        of each class.
    """
    # Number of classes
    classes = confusion.shape[0]
    # initialize precision array at 0
    precisions = np.zeros((classes,))

    for i in range(classes):
        # True positives for class i
        true_positives = confusion[i, i]
        # Sum elements in i-th column: predicted positive
        pred_pos = np.sum(confusion[:, i])
        # precision: ratio of TP over PP
        precisions[i] = true_positives / pred_pos

    return precisions
