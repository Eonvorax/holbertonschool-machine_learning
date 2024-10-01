#!/usr/bin/env python3
"""
Create masks
"""
import tensorflow as tf


def create_padding_mask(seq):
    """
    Creates a padding mask for the input sequence.
    The mask is a tensor of 0s and 1s, where 1 indicates a padding token.

    Args:
        seq: A tensor of shape `(batch_size, seq_len)` containing the
            input sequence.

    Returns:
        A mask tensor of shape `(batch_size, 1, 1, seq_len)`.
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    """
    Creates a look-ahead mask for the target sequence.
    The mask prevents the decoder from attending to future tokens.

    Args:
        size: The size of the mask `(seq_len_out)`.

    Returns:
        A mask tensor of shape `(seq_len_out, seq_len_out)`.
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_masks(inputs, target):
    """
    Creates all masks for training/validation.

    Args:
        inputs: A tf.Tensor of shape `(batch_size, seq_len_in)` containing
            the input sentence.
        target: A tf.Tensor of shape `(batch_size, seq_len_out)` containing
            the target sentence.

    Returns:
    - :encoder_mask: Padding mask for the encoder
        `(batch_size, 1, 1, seq_len_in)`.
    - :combined_mask: Combined look-ahead and padding mask for the first
        attention block in the decoder
        `(batch_size, 1, seq_len_out, seq_len_out)`.
    - :decoder_mask: Padding mask for the second attention block in the
        decoder `(batch_size, 1, 1, seq_len_in)`.
    """
    encoder_mask = create_padding_mask(inputs)
    decoder_mask = create_padding_mask(inputs)

    # Look-ahead mask for the target
    look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])

    # Target padding mask (for padding in the target sentence)
    dec_target_padding_mask = create_padding_mask(target)

    # Mask for 1st decoder attention block (look-ahead + target padding)
    combined_mask = tf.maximum(look_ahead_mask, dec_target_padding_mask)

    return encoder_mask, combined_mask, decoder_mask
