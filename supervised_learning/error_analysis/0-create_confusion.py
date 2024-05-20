#!/usr/bin/env python3
"""
Confusion matrix
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix.

    Args:
        - labels is a one-hot numpy.ndarray of shape (m, classes) containing
        the correct labels for each data point.
        - logits is a one-hot numpy.ndarray of shape (m, classes) containing
        the predicted labels.

    Returns:
        A confusion numpy.ndarray of shape (classes, classes) with row indices
        representing the correct labels and column indices representing
        the predicted labels.
    """
    # Decode one-hot encoded labels back to class indices
    true_labels = np.argmax(labels, axis=1)
    predicted_labels = np.argmax(logits, axis=1)

    # Number of classes
    classes = labels.shape[1]

    # Initialize elements of confusion matrix at zero
    confusion_matrix = np.zeros((classes, classes))

    # NOTE When the classifier is accurate, (true_label, predicted_label)
    # pairs will often be of the form (i, i) where i is the class index
    for true, pred in zip(true_labels, predicted_labels):
        # Increment at class indices (true, predicted)
        confusion_matrix[true, pred] += 1

    return confusion_matrix
