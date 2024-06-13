#!/usr/bin/env python3
"""
Transition Layer
"""

from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in
    'Densely Connected Convolutional Networks (2018)'

    Parameters:
    X (tf.Tensor): The output of the previous layer.
    nb_filters (int): The number of filters in X.
    compression (float): The compression factor for the transition layer.

    Returns:
    tf.Tensor: The output of the transition layer.
    int: The number of filters within the output.
    """
    # Reduce number of filters in output by the compression factor θ
    new_nb_filters = int(nb_filters * compression)

    # Initializer he_normal with seed 0
    init = K.initializers.HeNormal(seed=0)

    norm = K.layers.BatchNormalization()(X)
    activ = K.layers.Activation(activation="relu")(norm)

    # 1x1 "bottleneck" convolution, compressed by θ factor
    conv = K.layers.Conv2D(filters=new_nb_filters,
                           kernel_size=(1, 1),
                           padding="same",
                           kernel_initializer=init)(activ)

    # Average Pooling 2x2, stride 2
    avg_pool = K.layers.AvgPool2D(pool_size=(2, 2),
                                  strides=(2, 2),
                                  padding="same")(conv)

    return avg_pool, new_nb_filters
