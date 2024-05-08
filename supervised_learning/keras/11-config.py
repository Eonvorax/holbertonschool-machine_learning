#!/usr/bin/env python3
"""
This is the 11-config module.
"""

import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model's configuration in JSON format:

    network is the model whose configuration should be saved
    filename is the path of the file the configuration should be saved to

    Returns: None
    """
    network_config = network.to_json()

    with open(filename, 'w') as json_file:
        json_file.write(network_config)


def load_config(filename):
    """
    Loads a model with a specific configuration:

    filename is the path of the file containing the model's configuration in
    JSON format.

    Returns: the loaded model
    """
    with open(filename, 'r') as json_file:
        network_config = json_file.read()

    return K.models.model_from_json(network_config)
