#!/usr/bin/env python3

"""
This is the 4-train module.
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, verbose=True,
                shuffle=False):
    """
    trains a model using mini-batch gradient descent:

    - network is the model to train
    - data is a numpy.ndarray of shape (m, nx) containing the input data
    - labels is a one-hot numpy.ndarray of shape (m, classes) containing the
        labels of data
    - batch_size is the size of the batch for mini-batch gradient descent
    - epochs is the number of passes through data for radient descent
    - verbose is a boolean that determines if output should be printed
    - shuffle is a boolean that determines whether to shuffle the batches
        every epoch.

    Returns: the History object generated after training the model
    """
    # NOTE notation: x=data, y=labels
    return network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                       verbose=verbose, shuffle=shuffle)
