#!/usr/bin/env python3

"""
This is the 9-model module.
"""

import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire model.

    network is the model to save
    filename is the path of the file that the model should be saved to

    Returns: None
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    Loads an entire model.

    filename is the path of the file that the model should be loaded from

    Returns: the loaded model
    """
    return K.models.load_model(filename)
