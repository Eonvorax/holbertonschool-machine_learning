#!/usr/bin/env python3
"""
Vanilla Autoencoder
"""

import tensorflow.keras as keras


def build_encoder(input_dims, hidden_layers, latent_dims):
    """
    Builds the encoder part of the autoencoder.
    """
    # Define the input layer
    encoder_input = keras.layers.Input(shape=(input_dims,))

    # Build the encoder with the given hidden layers
    x = encoder_input
    for units in hidden_layers:
        x = keras.layers.Dense(units=units, activation='relu')(x)

    # Latent space representation (encoded)
    latent_space = keras.layers.Dense(latent_dims, activation='relu')(x)

    return keras.Model(inputs=encoder_input, outputs=latent_space)


def build_decoder(latent_dims, hidden_layers, output_dims):
    """
    Builds the decoder part of the autoencoder.
    """
    # Define the input layer for the latent space
    decoder_input = keras.layers.Input(shape=(latent_dims,))

    # Build the decoder with the given hidden layers, in reverse
    x = decoder_input
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units=units, activation='relu')(x)

    # Output layer with sigmoid activation (decoded)
    decoder_output = keras.layers.Dense(output_dims, activation='sigmoid')(x)

    return keras.models.Model(decoder_input, decoder_output)


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a "vanilla" autoencoder. The autoencoder will be compiled using
    Adam optimization and binary cross-entropy loss. All layers use relu
    activation, except the last layer in the decoder, which uses sigmoid
    instead.

    :Parameters:
    - input_dims is an integer containing the dimensions of the model input
    - hidden_layers is a list containing the number of nodes for each hidden
    layer in the encoder, respectively. The hidden layers should be reversed
    for the decoder
    - latent_dims is an integer containing the dimensions of the latent space
    representation

    :Returns:
    - A tuple with (encoder, decoder, auto):
        - encoder is the encoder model
        - decoder is the decoder model
        - auto is the full autoencoder model
    """
    encoder = build_encoder(input_dims, hidden_layers, latent_dims)
    decoder = build_decoder(latent_dims, hidden_layers, input_dims)

    encoder_input = keras.layers.Input(shape=(input_dims,))
    encoded_output = encoder(encoder_input)
    decoded_output = decoder(encoded_output)

    # Compile the full autoencoder model
    auto = keras.Model(inputs=encoder_input, outputs=decoded_output)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
