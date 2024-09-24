#!/usr/bin/env python3
"""
Self Attention
"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    This class represents the self-attention mechanism for machine translation
    """

    def __init__(self, units):
        """
        Initializes the SelfAttention layer.

        :Parameters:
        - units (int): The number of hidden units for the attention mechanism.

        Sets up three dense layers:
        - W: Applied to the previous decoder hidden state (s_prev).
        - U: Applied to the encoder hidden states (hidden_states).
        - V: Produces scalar score from sum of W(s_prev) and U(hidden_states).
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Forward pass to compute attention mechanism.

        :Parameters:
        s_prev : tf.Tensor:
            Tensor of shape (batch, units) containing the previous decoder
            hidden state.
        hidden_states : tf.Tensor
            Tensor of shape (batch, input_seq_len, units) containing
            the encoder hidden states.

        Returns:
        context (tf.Tensor):
            A tensor of shape (batch, units) that contains the context vector
            for the decoder.
        weights (tf.Tensor):
            A tensor of shape (batch, input_seq_len, 1) that contains the
            attention weights for each encoder hidden state.
        """
        # Expand dimensions of the previous hidden state for broadcasting
        s_prev_expanded = tf.expand_dims(input=s_prev, axis=1)

        # Comput scalar score for each time step, softmaxed to get weights
        scores = self.V(
            tf.nn.tanh(self.W(s_prev_expanded) + self.U(hidden_states)))
        weights = tf.nn.softmax(scores, axis=1)

        # Weighted sum of encoder hidden states (reduce over input_seq_len dim)
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
