#!/usr/bin/env python3
"""
Create a Layer with L2 Regularization.
"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a neural network layer with L2 regularization.

    Parameters:
    prev (tensor): tensor containing the output of the previous layer.
    n (int): number of nodes the new layer should contain.
    activation (function): activation function to use on the layer.
    lambtha (float): L2 regularization parameter.

    Returns:
    tensor: the output of the new layer.
    """
    # Weight initialization: He et. al
    init_weights = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                         mode=("fan_avg"))
    # Regularization for this layer's loss: L2 regularizer
    l2_regularizer = tf.keras.regularizers.L2(lambtha)

    # Create Dense layer with L2 regularization
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init_weights,
        kernel_regularizer=l2_regularizer
    )

    return layer(prev)
