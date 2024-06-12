#!/usr/bin/env python3
"""
DenseNet Block
"""

from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in
    'Densely Connected Convolutional Networks (2018)'

    Parameters:
        X (tf.Tensor): The output from the previous layer.
        nb_filters (int): The number of filters in X.
        growth_rate (int): The growth rate for the dense block.
        layers_count (int): The number of layers in the dense block.

    Returns:
        tf.Tensor: Concatenated output of each layer within the Dense Block.
        int: The number of filters within the concatenated outputs.
    """
    # Initializer he_normal with seed 0
    init = K.initializers.HeNormal(seed=0)

    for layer_i in range(layers):
        # Batch normalization and ReLU activation before convolution
        norm1 = K.layers.BatchNormalization()(X)
        activ1 = K.layers.Activation(activation="relu")(norm1)

        # 1x1 "bottleneck" convolution, with '4 x k' channels
        conv1 = K.layers.Conv2D(filters=4 * growth_rate,
                                kernel_size=(1, 1),
                                padding="same",
                                kernel_initializer=init)(activ1)

        # BatchNorm and ReLU, before 3x3 convolution again
        norm2 = K.layers.BatchNormalization()(conv1)
        activ2 = K.layers.Activation("relu")(norm2)
        conv2 = K.layers.Conv2D(filters=growth_rate,
                                kernel_size=(3, 3),
                                padding="same",
                                kernel_initializer=init)(activ2)

        # Concatenate inputs and new outputs on channel axis
        X = K.layers.Concatenate()([X, conv2])

        # Update the number of filters by the growth rate
        nb_filters += growth_rate

    return X, nb_filters
