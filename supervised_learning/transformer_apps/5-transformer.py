#!/usr/bin/env python3
"""
Full Transformer Network
"""

import numpy as np
import tensorflow as tf


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer.

    :param max_seq_len: an integer representing the maximum sequence length
    :param dm: the model depth

    Returns:
    A numpy.ndarray of shape `(max_seq_len, dm)` containing the
    positional encoding vectors
    """
    pos_encoding_vectors = np.zeros(shape=(max_seq_len, dm))
    # Loop over each position
    for pos in range(max_seq_len):
        # Loop over each dimension
        for i in range(0, dm // 2):
            # Compute scaling factor for the position
            div_term = 10000 ** (2 * i / dm)

            # Apply sin computation to the even indices (2i)
            pos_encoding_vectors[pos, 2*i] = np.sin(pos / div_term)

            # Apply cos function to the odd indices (2i + 1)
            pos_encoding_vectors[pos, 2*i + 1] = np.cos(pos / div_term)

    return pos_encoding_vectors


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
