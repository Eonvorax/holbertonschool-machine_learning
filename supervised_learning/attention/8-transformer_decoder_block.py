#!/usr/bin/env python3
"""
Transformer Decoder Block
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """
    This class represents a transformer's decoder block.
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Initializes the decoder block.

        :param dm: Integer representing the dimensionality of the model.
        :param h: Integer representing the number of attention heads.
        :param hidden: Integer representing the number of hidden units in the
        fully connected layer.
        :param drop_rate: Float representing the dropout rate.
        """
        super().__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Forward pass through the transformer's decoder block.

        :param x: a tensor of shape `(batch, target_seq_len, dm)` containing
        the input to the decoder block
        :param encoder_output: a tensor of shape `(batch, input_seq_len, dm)`
        containing the output of the encoder
        :param training: a boolean to determine if the model is training
        :param look_ahead_mask: the mask to be applied to the first multi head
        attention layer
        :param padding_mask: the mask to be applied to the second multi head
        attention layer

        Returns:
        A tensor of shape `(batch, target_seq_len, dm)` containing the block's
        output
        """
        # Masked Multi-head attention
        masked_mha_output, _ = self.mha1(x, x, x, look_ahead_mask)
        # 1st dropout
        masked_mha_output = self.dropout1(masked_mha_output, training=training)
        # 1st residual connection + layer normalization
        output1 = self.layernorm1(x + masked_mha_output)

        # Second multi-head attention
        mha2_output, _ = self.mha2(output1, encoder_output, encoder_output,
                                   padding_mask)
        mha2_output = self.dropout2(mha2_output)

        # 2nd residual connection + layer normalization
        output2 = self.layernorm2(mha2_output + output1)

        # Feed-forward neural network: 1st dense layer with ReLU activation
        ff_output = self.dense_hidden(output2)
        # Second dense layer
        ff_output = self.dense_output(ff_output)
        ff_output = self.dropout3(ff_output, training=training)

        # 2nd Residual connection + layer normalization
        output2 = self.layernorm3(ff_output + output2)

        return output2
