#!/usr/bin/env python3
"""
Calculate specificity
"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): confusion matrix, shape (classes, classes)
        where row indices represent the correct labels and column indices
        represent the predicted labels.

    Returns:
        numpy.ndarray: An array of shape (classes,) containing the specificity
        of each class.
    """
    # Number of classes
    classes = confusion.shape[0]
    # initialize specificity array at 0
    specificities = np.zeros((classes,))

    for i in range(classes):
        # True positives for class i
        true_pos = confusion[i, i]
        # False positives for class i: subtract true positives from column i
        false_pos = np.sum(confusion[:, i]) - true_pos
        # False negatives for class i: subtract true positives from row i
        false_neg = np.sum(confusion[i, :]) - true_pos
        # True negatives for class i: subtract to leave only true negatives
        true_neg = np.sum(confusion) - (true_pos + false_pos + false_neg)

        # Specificity: ratio of TN over TN + FP
        specificities[i] = true_neg / (true_neg + false_pos)

    return specificities
