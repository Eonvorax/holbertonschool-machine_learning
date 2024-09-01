#!/usr/bin/env python3

"""
Generating faces
"""

from tensorflow import keras


def convolutional_GenDiscr():
    """
    Builds a convolutional generator and discriminator with the functional API.

    Returns:
        - A generator model that maps a latent vector of shape (16) to an
        output of shape (16, 16, 1).
        - A discriminator model that maps an input of shape (16, 16, 1) to a
        single output (probability).
    """

    def build_gen_block(x, filters):
        """
        Builds a block for the generator model.

        Args:
            x: The input tensor.
            filters: Number of filters for the Conv2D layer.

        Returns:
            - The output tensor after applying UpSampling2D, Conv2D,
            BatchNormalization, and Activation layers.
        """
        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('tanh')(x)
        return x

    def build_discr_block(x, filters):
        """
        Builds a block for the discriminator model.

        Args:
            x: The input tensor.
            filters: Number of filters for the Conv2D layer.

        Returns:
            - The output tensor after applying Conv2D, MaxPooling2D,
            and Activation layers.
        """
        x = keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation('tanh')(x)
        return x

    def get_generator():
        """
        Builds the generator model using the functional API.

        Returns:
            - A generator model.
        """
        inputs = keras.Input(shape=(16,))
        x = keras.layers.Dense(2048, activation='tanh')(inputs)
        x = keras.layers.Reshape((2, 2, 512))(x)

        # Apply 3 generator blocks with decreasing filters
        x = build_gen_block(x, 64)
        x = build_gen_block(x, 16)
        x = build_gen_block(x, 1)

        # Create the generator model
        return keras.Model(inputs, x, name='generator')

    def get_discriminator():
        """
        Builds the discriminator model using the functional API.

        Returns:
            - A discriminator model.
        """
        inputs = keras.Input(shape=(16, 16, 1))

        # Apply 4 discriminator blocks with increasing filters
        x = build_discr_block(inputs, 32)
        x = build_discr_block(x, 64)
        x = build_discr_block(x, 128)
        x = build_discr_block(x, 256)

        # Flatten and tanh-activated Dense layer for final output
        x = keras.layers.Flatten()(x)
        outputs = keras.layers.Dense(1, activation='tanh')(x)

        # Create the discriminator model
        return keras.Model(inputs, outputs, name='discriminator')

    return get_generator(), get_discriminator()
