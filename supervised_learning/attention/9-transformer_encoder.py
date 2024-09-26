#!/usr/bin/env python3
"""
Transformer Encoder
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """
    This class represents an transformer's Encoder.
    """
    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Initializes the Encoder.

        :param N: Integer representing the number of encoder blocks.
        :param dm: Integer representing the dimensionality of the model.
        :param h: Integer representing the number of attention heads.
        :param hidden: Integer representing the number of hidden units in
        the fully connected layer.
        :param input_vocab: Integer representing the size of the input
        vocabulary.
        :param max_seq_len: Integer representing the maximum sequence length
        possible.
        :param drop_rate: Float representing the dropout rate.
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_dim=input_vocab,
                                                   output_dim=dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Forward pass through the `Encoder`.

        :param x: Tensor of shape `(batch, input_seq_len)` containing the input
        to the encoder (tokenized input).
        :param training: Boolean to determine if the model is training.
        :param mask: Mask to be applied for multi-head attention.

        Returns:
        A tensor of shape `(batch, input_seq_len, dm)` containing the encoder
        output.
        """
        input_seq_len = x.shape[1]

        # embedding; new shape: (batch, input_seq_len, dm)
        x = self.embedding(x)

        # positional encoding, scaled by sqrt(dm)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:input_seq_len, :]

        # Apply dropout to the positional encoding
        x = self.dropout(x, training=training)

        # Pass the input through each encoder block
        for i in range(self.N):
            x = self.blocks[i](x, training, mask)

        return x
