#!/usr/bin/env python3
"""
Positional Encoding
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer.

    :param max_seq_len: an integer representing the maximum sequence length
    :param dm: the model depth

    Returns:
    A numpy.ndarray of shape `(max_seq_len, dm)` containing the
    positional encoding vectors
    """
    pos_encoding_vectors = np.zeros(shape=(max_seq_len, dm))
    # Loop over each position
    for pos in range(max_seq_len):
        # Loop over each dimension
        for i in range(0, dm // 2):
            # Compute scaling factor for the position
            div_term = 10000 ** (2 * i / dm)

            # Apply sin computation to the even indices (2i)
            pos_encoding_vectors[pos, 2*i] = np.sin(pos / div_term)

            # Apply cos function to the odd indices (2i + 1)
            pos_encoding_vectors[pos, 2*i + 1] = np.cos(pos / div_term)

    return pos_encoding_vectors
