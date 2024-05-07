#!/usr/bin/env python3

"""
This is the 6-train module.
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent:

    - network is the model to train
    - data is a numpy.ndarray of shape (m, nx) containing the input data
    - labels is a one-hot numpy.ndarray of shape (m, classes) containing the
        labels of data
    - batch_size is the size of the batch for mini-batch gradient descent
    - epochs is the number of passes through data for radient descent
    - verbose is a boolean that determines if output should be printed
    - shuffle is a boolean that determines whether to shuffle the batches
        every epoch.
    - validation_data is the data to validate the model with, if not None
    - patience is the number of epochs with no improvement after which
        training will be stopped.

    Returns: the History object generated after training the model
    """
    callback = []
    if early_stopping is True and validation_data is not None:
        # NOTE monitor defaults to "val_loss" anyway
        callback = K.callbacks.EarlyStopping(monitor='val_loss',
                                             patience=patience)

    # NOTE notation: x=data, y=labels
    return network.fit(x=data, y=labels, batch_size=batch_size, epochs=epochs,
                       verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data, callbacks=[callback])
