#!/usr/bin/env python3
"""
Training a full Transformer network
"""

import tensorflow as tf
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    """
    data = Dataset(batch_size, max_len)
    data_train = data.data_train
    data_valid = data.data_valid

    model = Transformer(N, dm, h, hidden,
                        input_vocab=data.tokenizer_pt.vocab_size,
                        target_vocab=data.tokenizer_en.vocab_size,
                        max_seq_input=max_len,
                        max_seq_target=max_len,
                        drop_rate=0.1)

    # TODO prepare optimizer, loss function, custom learning rate objects
