#!/usr/bin/env python3
"""
Calculate the softmax cross-entropy loss of a prediction.
"""

import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss of a prediction.

    Arguments:
    y: placeholder for the labels of the input data
    y_pred: tensor containing the network's predictions

    Returns:
    tensor containing the loss of the prediction
    """
    return tf.losses.softmax_cross_entropy(onehot_labels=y,
                                           logits=y_pred)
