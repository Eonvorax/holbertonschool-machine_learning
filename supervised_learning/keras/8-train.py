#!/usr/bin/env python3

"""
This is the 8-train module.
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False, save_best=False, filepath=None):
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
    - learning_rate_decay is a boolean that indicates whether learning rate
        decay should be used.
    - save_best is a boolean indicating if we save the model after each epoch
    - filepath is the file path where the model should be saved

    Returns: the History object generated after training the model
    """
    callbacks = []
    if early_stopping is True and validation_data is not None:
        # NOTE monitor defaults to "val_loss" anyway
        callbacks.append(K.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=patience))

    if learning_rate_decay is True and validation_data is not None:
        def scheduler(epochs):
            """Scheduler to callback, adjusts learning rate"""
            return alpha / (1 + decay_rate * epochs)

        callbacks.append(K.callbacks.LearningRateScheduler(
            scheduler, verbose=1))

    if save_best is True and filepath is not None:
        callbacks.append(K.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            mode='min',
            save_best_only=True))

    return network.fit(x=data, y=labels, batch_size=batch_size, epochs=epochs,
                       verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data, callbacks=[callbacks])
