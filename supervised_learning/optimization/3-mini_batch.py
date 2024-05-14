#!/usr/bin/env python3
"""
Creates mini-batches for training a neural network with mini-batch
gradient descent.
"""

import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches to be used for training a neural network using
    mini-batch gradient descent.

    Args:
        X: numpy.ndarray of shape (m, nx) representing input data
           m: number of data points
           nx: number of features in X
        Y: numpy.ndarray of shape (m, ny) representing the labels
           m: same number of data points as in X
           ny: number of classes for classification tasks.
        batch_size: number of data points in a batch

    Returns:
        List of mini-batches containing tuples (X_batch, Y_batch).
    """
    # NOTE Can also be done with np.array_split, this way is more explicit
    m = X.shape[0]
    mini_batches = []

    # Shuffle input data
    shuffled_X, shuffled_Y = shuffle_data(X, Y)

    # Calculate number of complete batches
    num_complete_batches = m // batch_size

    # Extract each mini-batch slice from the shuffled inputs
    for i in range(num_complete_batches):
        # Calc. indexes for this slice, then slice
        start = i * batch_size
        end = start + batch_size
        mini_batch_X = shuffled_X[start:end]
        mini_batch_Y = shuffled_Y[start:end]
        # Add resulting mini_batch to mini-batch tuple list
        mini_batches.append((mini_batch_X, mini_batch_Y))

    # If there are remaining data points, create a mini-batch with them
    if m % batch_size != 0:
        start = num_complete_batches * batch_size
        mini_batch_X = shuffled_X[start:]
        mini_batch_Y = shuffled_Y[start:]
        mini_batches.append((mini_batch_X, mini_batch_Y))

    return mini_batches
