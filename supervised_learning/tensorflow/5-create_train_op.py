#!/usr/bin/env python3
"""
Creates the training operation for the network.
"""

import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """
    Creates the training operation for the network using gradient descent.

    Arguments:
    loss: the loss of the network's prediction
    alpha: the learning rate

    Returns:
    an operation that trains the network using gradient descent
    """
    gradient_descent = tf.train.GradientDescentOptimizer(learning_rate=alpha)

    return gradient_descent.minimize(loss)
