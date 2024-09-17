#!/usr/bin/env python3
"""
Extract & convert Word2Vec model to a keras layer
"""
import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a gensim Word2Vec model to a keras Embedding layer

    :Parameters:
    - `model`: a trained gensim Word2Vec model

    Returns:
        The trainable keras Embedding
    """
    keyed_vectors = model.wv  # structure holding the result of training
    weights = keyed_vectors.vectors  # vectors themselves, a 2D numpy array

    layer = tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True
    )
    return layer
