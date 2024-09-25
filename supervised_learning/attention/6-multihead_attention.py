#!/usr/bin/env python3
"""
Multi Head Attention
"""

import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    This class represents the multi head attention mechanism.
    """

    def __init__(self, dm, h):
        """
        Initializes the MultiHeadAttention layer.

        :param dm: Integer representing the dimensionality of the model.
        :param h: Integer representing the number of attention heads.
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """
        Splits the input into multiple heads for multi-head attention.

        :param x: Tensor of shape `(batch_size, seq_len, dm)`

        Returns:
        Tensor of shape `(batch_size, h, seq_len, depth)`
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Computes the multi-head attention.

        :param Q: Tensor of shape `(batch_size, seq_len_q, dk)`
        :param K: Tensor of shape `(batch_size, seq_len_v, dk)`
        :param V: Tensor of shape `(batch_size, seq_len_v, dv)`
        :param mask: Mask tensor (currently not used)

        Returns: outputs, weights
            - :output: Tensor of shape `(batch_size, seq_len_q, dm)`
            - :weights: Tensor of shape `(batch_size, h, seq_len_q, seq_len_v)`
        """
        batch_size = tf.shape(Q)[0]

        # Generate queries, keys, and values by passing through Dense layers
        # NOTE Shape: (batch_size, seq_len_q, dm)
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        # Split Q, K, V into multiple heads
        # NOTE new shape: (batch_size, h, seq_len_q, depth)
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Apply scaled dot-product attention
        attention_output, weights = sdp_attention(Q, K, V, mask)

        # Concatenate heads (reversing the head-level split)
        # NOTE Shape (batch_size, seq_len_q, h, depth)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention_output, (batch_size, -1, self.dm))

        # Final linear layer: Shape (batch_size, seq_len_q, dm)
        output = self.linear(concat_attention)

        return output, weights
