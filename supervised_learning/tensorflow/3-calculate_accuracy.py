#!/usr/bin/env python3
"""
Calculate the accuracy of the prediction.
"""

import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction.

    Arguments:
    y: placeholder for the labels of the input data
    y_pred: tensor containing the network's predictions

    Returns:
    tensor containing the decimal accuracy of the prediction
    """
    # One-hot decoding y_pred & y to integer labels
    y_pred_labels = tf.argmax(y_pred, axis=1)
    y_true_labels = tf.argmax(y, axis=1)

    # Compare predicted labels with true labels, get a boolean tensor
    correct_preds = tf.equal(y_pred_labels, y_true_labels)

    # NOTE The boolean tensor is first converted to a float tensor
    # where True becomes 1.0 and False becomes 0.0
    # Returning the mean of the float values in the tensor
    return tf.reduce_mean(tf.cast(correct_preds, tf.float32))
