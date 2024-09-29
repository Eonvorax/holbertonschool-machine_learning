#!/usr/bin/env python3
"""
Full Transformer Network
"""
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """
    This class represents a complete transformer network.
    """

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
        Initializes the Transformer model.

        :param N: Integer representing the number of blocks in the encoder
        and decoder.
        :param dm: Integer representing the dimensionality of the model.
        :param h: Integer representing the number of attention heads.
        :param hidden: Integer representing the number of hidden units in
        the feed-forward layers.
        :param input_vocab: Integer representing the size of the input
        vocabulary.
        :param target_vocab: Integer representing the size of the target
        vocabulary.
        :param max_seq_input: Integer representing the maximum sequence length
        possible for the input.
        :param max_seq_target: Integer representing the maximum sequence length
        possible for the target.
        :param drop_rate: Float representing the dropout rate.
        """
        super().__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask,
             decoder_mask):
        """
        Forward pass through the Transformer network.

        :param inputs: Tensor of shape (batch, input_seq_len) containing the
            input sequence.
        :param target: Tensor of shape (batch, target_seq_len) containing the
            target sequence.
        :param training: Boolean to determine if the model is training.
        :param encoder_mask: The padding mask to be applied to the encoder.
        :param look_ahead_mask: The look ahead mask to be applied to the
        decoder.
        :param decoder_mask: The padding mask to be applied to the decoder.

        Returns:
        A tensor of shape (batch, target_seq_len, target_vocab) containing
        the transformer output.
        """
        # Pass inputs through the encoder
        enc_output = self.encoder(inputs, training, encoder_mask)

        # Pass the target and encoder output through the decoder
        dec_output = self.decoder(target, enc_output, training,
                                  look_ahead_mask, decoder_mask)

        # Pass the decoder output through the final linear layer
        final_output = self.linear(dec_output)

        return final_output
