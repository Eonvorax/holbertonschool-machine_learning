#!/usr/bin/env python3
"""
Inception Block
"""

from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block as described in the study
    'Going Deeper with Convolutions' (2014).

    Parameters:
    A_prev:
        output from the previous layer, with shape
        (batch_size, height, width, channels).
    filters:
        tuple or list containing the number of filters for each
    convolution in the inception block:
        F1: number of filters in the 1x1 convolution
        F3R: number of filters in the 1x1 convolution before 3x3 convolution
        F3: number of filters in the 3x3 convolution
        F5R: number of filters in the 1x1 convolution before 5x5 convolution
        F5: number of filters in the 5x5 convolution
        FPP: number of filters in the 1x1 convolution after the max pooling
        after the max pooling.

    Returns:
        Concatenated output of the inception block, with shape
        (batch_size, height, width, total_filters),
        where total_filters is the sum of F1, F3, F5, and FPP.
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    # Branch 1: 1x1 convolution branch
    conv1x1 = K.layers.Conv2D(F1, kernel_size=(1, 1), padding='same',
                              activation='relu')(A_prev)

    # Branch 2: 1x1 convolution before 3x3 convolution
    conv3x3_reduce = K.layers.Conv2D(F3R, kernel_size=(1, 1), padding='same',
                                     activation='relu')(A_prev)
    conv3x3 = K.layers.Conv2D(F3, kernel_size=(3, 3), padding='same',
                              activation='relu')(conv3x3_reduce)

    # Branch 3: 1x1 convolution before 5x5 convolution
    conv5x5_reduce = K.layers.Conv2D(F5R, kernel_size=(1, 1), padding='same',
                                     activation='relu')(A_prev)
    conv5x5 = K.layers.Conv2D(F5, kernel_size=(5, 5), padding='same',
                              activation='relu')(conv5x5_reduce)

    # Branch 4: 3x3 max pooling before 1x1 convolution
    maxpool = K.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1),
                                 padding='same')(A_prev)
    maxpool_conv = K.layers.Conv2D(FPP, kernel_size=(1, 1), padding='same',
                                   activation='relu')(maxpool)

    # Concatenate the branches outputs along the channels dimension
    output = K.layers.Concatenate(
        axis=-1)([conv1x1, conv3x3, conv5x5, maxpool_conv])
    # NOTE axis=-1 is equivalent to axis=3 here (hits last index dimension)

    return output
