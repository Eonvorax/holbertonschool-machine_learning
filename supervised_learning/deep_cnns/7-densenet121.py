#!/usr/bin/env python3
"""
DenseNet-121
"""

from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture as described in
    'Densely Connected Convolutional Networks (2018)'.

    Parameters:
    growth_rate (int): The growth rate for the dense blocks.
    compression (float): The compression factor for the transition layers.

    Returns:
    keras.Model: The Keras model of DenseNet-121.

    The model assumes the input data has shape (224, 224, 3). All convolutions
    are preceded by Batch Normalization and a rectified linear
    activation (ReLU), respectively.
    All weights use He normal initialization with seed set to zero.
    """
    # Initializer he_normal with seed 0
    init = K.initializers.HeNormal(seed=0)

    # input tensor (assuming given shape of data)
    input_data = K.Input(shape=(224, 224, 3))

    # First BN-ReLU-Conv, with twice the growth rate for initial filter number
    nb_filters = growth_rate * 2
    norm = K.layers.BatchNormalization()(input_data)
    activ = K.layers.Activation(activation="relu")(norm)
    conv = K.layers.Conv2D(filters=nb_filters,
                           kernel_size=(7, 7),
                           strides=(2, 2),
                           padding="same",
                           kernel_initializer=init)(activ)

    max_pool = K.layers.MaxPool2D(pool_size=(3, 3),
                                  strides=(2, 2),
                                  padding="same")(conv)

    # Dense block 1, transition layer 1, and so on until block 4
    # NOTE nb_filters is updated (halved) by transition_layer
    block1, nb_filters = dense_block(max_pool, nb_filters, growth_rate, 6)
    trans1, nb_filters = transition_layer(block1, nb_filters, compression)

    block2, nb_filters = dense_block(trans1, nb_filters, growth_rate, 12)
    trans2, nb_filters = transition_layer(block2, nb_filters, compression)

    block3, nb_filters = dense_block(trans2, nb_filters, growth_rate, 24)
    trans3, nb_filters = transition_layer(block3, nb_filters, compression)

    block4, nb_filters = dense_block(trans3, nb_filters, growth_rate, 16)

    # Average Pooling (7x7 global)
    avg_pool = K.layers.AvgPool2D(pool_size=(7, 7), strides=(1, 1))(block4)

    # Fully Connected Layer, softmax
    dense_softmax = K.layers.Dense(units=1000, activation='softmax',
                                   kernel_initializer=init)(avg_pool)

    return K.Model(inputs=input_data, outputs=dense_softmax)
