#!/usr/bin/env python3
"""
Sets up gradient descent momentum optimization in TensorFlow.
"""

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Sets up the gradient descent with momentum optimization algorithm
    in TensorFlow.

    Args:
        alpha (float): The learning rate.
        beta1 (float): The momentum weight.

    Returns:
        tf.keras.optimizers.Optimizer: Optimizer for gradient descent
        with momentum.
    """
    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)

    return optimizer
