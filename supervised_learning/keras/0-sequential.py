#!/usr/bin/env python3

"""
This is the 0-sequential module.
"""

from keras.src.models import Sequential
from keras.src.layers import Dense, Dropout
from keras import regularizers


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
    model = Sequential()

    for i, _ in enumerate(layers):
        model.add(Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=regularizers.L2(lambtha),
            input_shape=(nx,)
            ))

        # Add dropout layer on every layer except the last (output) layer
        if i != len(layers) - 1 and keep_prob is not None:
            model.add(Dropout(1 - keep_prob))

    return model
