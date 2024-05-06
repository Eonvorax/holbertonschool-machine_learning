#!/usr/bin/env python3

"""
This is the 2-optimize module.
"""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Sets up Adam optimization for a keras model with categorical
    crossentropy loss and accuracy metrics:

    network is the model to optimize
    alpha is the learning rate
    beta1 is the first Adam optimization parameter
    beta2 is the second Adam optimization parameter

    Returns: None
    """
    opt_adam = K.optimizers.Adam(learning_rate=alpha,
                          beta_1=beta1,
                          beta_2=beta2)

    network.compile(optimizer=opt_adam,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    return None
