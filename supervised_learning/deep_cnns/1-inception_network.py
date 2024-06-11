#!/usr/bin/env python3
"""
Inception Network
"""

from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Builds the inception network as described in
    'Going Deeper with Convolutions' (2014).

    This function constructs an Inception network with the following
    specifications:
    - Input shape: (224, 224, 3)
    - All convolutions (including the inception block) use ReLU activation.
    - Uses the inception_block function for constructing inception modules.

    Returns:
    tf.keras.Model
        The Keras model of the inception network.
    """

    # input tensor (assuming given shape)
    input_data = K.Input(shape=(224, 224, 3))

    conv1 = K.layers.Conv2D(filters=64,
                            kernel_size=(7, 7),
                            strides=(2, 2),
                            padding="same",
                            activation="relu")(input_data)

    maxpool_1 = K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                                   padding="same")(conv1)

    # 2-depth conv. like inception block: reduce dims then convolve again
    conv2 = K.layers.Conv2D(filters=64,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            padding="same",
                            activation="relu")(maxpool_1)
    conv3 = K.layers.Conv2D(filters=192,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding="same",
                            activation="relu")(conv2)

    maxpool_2 = K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                                   padding="same")(conv3)

    # 2 inception blocks
    inception_3a = inception_block(maxpool_2, [64, 96, 128, 16, 32, 32])
    inception_3b = inception_block(inception_3a, [128, 128, 192, 32, 96, 64])

    maxpool_3 = K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                                   padding="same")(inception_3b)

    # 5 inception blocks in a row
    inception_4a = inception_block(maxpool_3, [192, 96, 208, 16, 48, 64])
    inception_4b = inception_block(inception_4a, [160, 112, 224, 24, 64, 64])
    inception_4c = inception_block(inception_4b, [128, 128, 256, 24, 64, 64])
    inception_4d = inception_block(inception_4c, [112, 144, 288, 32, 64, 64])
    inception_4e = inception_block(inception_4d, [256, 160, 320, 32, 128, 128])

    maxpool_4 = K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                                   padding="same")(inception_4e)

    # 2 inception blocks in a row
    inception_5a = inception_block(maxpool_4, [256, 160, 320, 32, 128, 128])
    inception_5b = inception_block(inception_5a, [384, 192, 384, 48, 128, 128])

    avg_pool = K.layers.AvgPool2D(pool_size=(7, 7),
                                  strides=(1, 1))(inception_5b)

    dropout = K.layers.Dropout(rate=0.4)(avg_pool)

    outputs = K.layers.Dense(units=1000, activation="softmax")(dropout)

    return K.Model(inputs=input_data, outputs=outputs)
