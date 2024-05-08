#!/usr/bin/env python3
"""
This is the 13-predict module.
"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network:

    network is the network model to make the prediction with
    data is the input data to make the prediction with
    verbose is a boolean that determines if output should be printed
    during the prediction process

    Returns: the prediction for the data
    """
    return network.predict(x=data, verbose=verbose)
