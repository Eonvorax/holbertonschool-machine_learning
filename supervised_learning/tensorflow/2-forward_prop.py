#!/usr/bin/env python3
"""
Create a tensor layer.
"""

import tensorflow.compat.v1 as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network.

    Arguments:
    x: placeholder for the input data
    layer_sizes: list containing the number of nodes in each layer
    activations: list containing the activation functions for each layer

    Returns:
    prediction of the network in tensor form
    """
    # First input is the placeholder x
    next_input = x

    for size, activ in zip(layer_sizes, activations):
        # Updating input with the output of the layer
        next_input = create_layer(next_input, size, activ)

    # Returning the prediction: output of the last layer
    return next_input
