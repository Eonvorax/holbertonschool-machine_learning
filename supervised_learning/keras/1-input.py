#!/usr/bin/env python3

"""
This is the 1-input module.
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with Keras.

    - nx is the number of input features to the network
    - layers is a list containing the number of nodes in each layer of
    the network
    - activations is a list containing the activation functions used for
    each layer of the network
    - lambtha is the L2 regularization parameter
    - keep_prob is the probability that a node will be kept for dropout

    Returns: the keras model
    """
    inputs = K.Input(shape=(nx,))
    # x starts as the input tensor to the model
    x = inputs

    for i, layer in enumerate(layers):
        x = K.layers.Dense(
            layer,
            activation=activations[i],
            kernel_regularizer=K.regularizers.L2(lambtha)
        )(x)  # passing x through this dense layer and updating it

        # Add dropout layer on every layer except the last (output) layer
        if i != len(layers) - 1 and keep_prob is not None:
            # Passing x through Dropout layer and updating it
            x = (K.layers.Dropout(1 - keep_prob))(x)

    # Build Model using predefined inputs and outputs
    model = K.Model(inputs=inputs, outputs=x)
    return model
