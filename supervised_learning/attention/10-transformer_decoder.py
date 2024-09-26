#!/usr/bin/env python3
"""
Transformer Decoder
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """
    This class represents an transformer's Decoder.
    """

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Initializes the Decoder.

        :param N: the number of blocks in the encoder
        :param dm: the dimensionality of the model
        :param h: the number of heads
        :param hidden: the number of hidden units in the fully connected layer
        :param target_vocab: the size of the target vocabulary
        :param max_seq_len: the maximum sequence length possible
        :param drop_rate: the dropout rate
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_dim=target_vocab,
                                                   output_dim=dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Forward pass through the `Decoder`.

        :param x: Tensor of shape `(batch, input_seq_len)` containing the input
        to the decoder (tokenized input).
        :param encoder_output: Tensor of shape `(batch, input_seq_len, dm)`
        containing the output of the encoder
        :param training: Boolean to determine if the model is training.
        :param look_ahead_mask: Mask to be applied to the first multi head
        attention layer
        :param padding_mask: Mask to be applied to the second multi head
        attention layer

        Returns:
        A tensor of shape `(batch, input_seq_len, dm)` containing the decoder
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

        # Pass the input through each decoder block
        for i in range(self.N):
            x = self.blocks[i](x, encoder_output, training, look_ahead_mask,
                               padding_mask)

        return x
