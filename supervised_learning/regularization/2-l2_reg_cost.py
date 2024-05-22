#!/usr/bin/env python3
"""
L2 Regularization Cost.
"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization.

    Parameters:
    cost (tensor): a tensor containing the cost of the network without
        L2 regularization.
    model (tf.keras.Model): a Keras model that includes layers with L2
        regularization.

    Returns:
    tensor: a tensor containing the total cost for each layer of the network,
        accounting for L2 regularization.
    """
    return cost + model.losses
