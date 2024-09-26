#!/usr/bin/env python3
"""
Transformer Encoder Block
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    This class represents a transformer's encoder block.
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Initializes the encoder block.

        :param dm: Integer representing the dimensionality of the model.
        :param h: Integer representing the number of attention heads.
        :param hidden: Integer representing the number of hidden units in the
        fully connected layer.
        :param drop_rate: Float representing the dropout rate.
        """
        super().__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        Forward pass through the encoder block.

        :param x: Tensor of shape `(batch, input_seq_len, dm)` containing the
        input to the encoder block.
        :param training: Boolean indicating whether the model is in training
        mode.
        :param mask: Mask to be applied for multi-head attention (optional).

        Returns:
        A tensor of shape `(batch, input_seq_len, dm)` containing the block's
        output.
        """
        # Multi-head attention
        mha_output, _ = self.mha(x, x, x, mask)
        # 1st dropout
        mha_output = self.dropout1(mha_output, training=training)
        # Residual connection + layer normalization
        output1 = self.layernorm1(x + mha_output)

        # Feed-forward neural network: 1st dense layer with ReLU activation
        ff_output = self.dense_hidden(output1)
        # Second dense layer
        ff_output = self.dense_output(ff_output)
        ff_output = self.dropout2(ff_output, training=training)

        # 2nd Residual connection + layer normalization
        output2 = self.layernorm2(output1 + ff_output)

        return output2
