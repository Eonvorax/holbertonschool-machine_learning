#!/usr/bin/env python3
"""
Create a tensor layer.
"""

import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    Creates a layer for a neural network.

    Arguments:
    prev: tensor output of the previous layer
    n: number of nodes in the layer to create
    activation: activation function that the layer should use

    Returns:
    tensor output of the layer
    """
    # Initialize the weights using He et al. (this line is given)
    init_weights = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # Create layer, named "layer", with n units, using init_weigths
    layer = tf.keras.layers.Dense(units=n, activation=activation,
                                  kernel_initializer=init_weights,
                                  name="layer")

    # Apply layer to input tensor (basically: perform forward propagation)
    return layer(prev)
