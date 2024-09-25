#!/usr/bin/env python3
"""
Scaled dot product Attention
"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calculates the scaled dot product attention.
    (The preceding dimensions of Q, K, and V are the same)

    :param Q: A tensor with its last two dimensions as `(..., seq_len_q, dk)`
        containing the query matrix.
    :param K: A tensor with its last two dimensions as `(..., seq_len_v, dk)`
        containing the key matrix.
    :param V: A tensor with its last two dimensions as `(..., seq_len_v, dv)`
        containing the value matrix.
    :param (Optional) mask: A tensor that can be broadcast into
        `(..., seq_len_q, seq_len_v)` containing the optional mask, or defaults
        to `None`. If mask is not `None`, multiply `-1e9` to the mask and add
        it to the scaled matrix multiplication.

    Returns: output, weights
        - :output: a tensor with its last two dimensions as
            `(..., seq_len_q, dv)` containing the scaled dot product attention
        - :weights: a tensor with its last two dimensions as
            `(..., seq_len_q, seq_len_v)` containing the attention weights
    """
    # Get dk from last dimension of Q (or K)
    # NOTE cast it to float32 to avoid type issues with tf
    dk = tf.cast(Q.shape[-1], dtype=tf.float32)

    # Matrix multiplication of Q by transposed K
    scores = tf.matmul(Q, K, transpose_b=True)

    # Scale attention scores by square root of dk
    scaled_scores = scores / tf.sqrt(dk)

    # If given a mask, apply it to the scores
    if mask:
        # Masked positions basically get set to a large "-inf" negative value
        scaled_scores += (mask * -1e-9)

    # Softmax gets us the attention weights
    attention_weights = tf.nn.softmax(scaled_scores, axis=-1)

    # Multiply the weights by the values V to get the output
    output = tf.matmul(attention_weights, V)

    return output, attention_weights
