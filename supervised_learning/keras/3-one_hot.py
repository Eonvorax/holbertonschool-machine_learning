#!/usr/bin/env python3

"""
This is the 3-one_hot module.
"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix:

    The last dimension of the one-hot matrix must be the number of classes

    Returns: the one-hot matrix
    """
    return K.utils.to_categorical(x=labels, num_classes=classes)
