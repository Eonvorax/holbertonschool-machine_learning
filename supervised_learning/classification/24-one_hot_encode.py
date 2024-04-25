#!/usr/bin/env python3

"""
This is the 24-one_hot_encode module.
"""

import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix.

    Y is a numpy.ndarray with shape (m,) containing numeric class labels
    m is the number of examples
    classes is the maximum number of classes found in Y

    Returns:
        A one-hot encoding of Y with shape (classes, m),
        or None on failure
    """
    if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
        return None
    if not isinstance(classes, int) or 0 <= classes <= np.max(Y):
        return None

    # NOTE neat trick, ex. with Y = [5 0 4 1 9 2 1 3 1 4]
    # Building matrix with row index 5 of identity matrix, then 0, then 4...
    # Then transpose so the shape is (classes, m), and not (m, classes)
    return np.eye(classes)[Y].T
