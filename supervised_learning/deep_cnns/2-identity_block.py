#!/usr/bin/env python3
"""
Identity Block
"""

from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in
    'Deep Residual Learning for Image Recognition' (2015).

    Parameters:
    A_prev : tensor
        The output of the previous layer.
    filters : tuple or list
        Contains F11, F3, F12 respectively:
            F11 : int
                Number of filters in the first 1x1 convolution.
            F3 : int
                Number of filters in the 3x3 convolution.
            F12 : int
                Number of filters in the second 1x1 convolution.

    Returns:
    tensor
        The activated output of the identity block.
    """
    F11, F3, F12 = filters

    # Initializer he_normal with seed 0
    init = K.initializers.HeNormal(seed=0)

    # First layer of left branch
    conv1 = K.layers.Conv2D(filters=F11,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            padding="same",
                            kernel_initializer=init)(A_prev)

    norm1 = K.layers.BatchNormalization(axis=-1)(conv1)
    relu1 = K.layers.Activation(activation="relu")(norm1)
    # NOTE could also use layers.ReLU() directly instead

    # Second layer of left branch
    conv2 = K.layers.Conv2D(filters=F3,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding="same",
                            kernel_initializer=init)(relu1)
    norm2 = K.layers.BatchNormalization(axis=-1)(conv2)
    relu2 = K.layers.Activation(activation="relu")(norm2)

    # Final layer of left branch
    conv3 = K.layers.Conv2D(filters=F12,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            padding="same",
                            kernel_initializer=init)(relu2)
    norm3 = K.layers.BatchNormalization(axis=-1)(conv3)

    # Merge output of left branch and right branch (input A_prev)
    merged = K.layers.Add()([norm3, A_prev])

    # Return activated output of merge, using ReLU
    return K.layers.Activation(activation="relu")(merged)
