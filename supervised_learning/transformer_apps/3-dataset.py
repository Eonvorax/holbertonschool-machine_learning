#!/usr/bin/env python3
"""
Pipeline
"""

import tensorflow_datasets as tfds
import transformers
import tensorflow as tf


class Dataset:
    """
    A class to load and prepare the TED HRLR translation dataset
    for machine translation from Portuguese to English.
    """

    def __init__(self, batch_size, max_len):
        """
        Initializes the Dataset object and loads the training and validation
        datasets.
        Also initializes tokenizers for Portuguese and English.
        Tokenizes the training and validation data by mapping the encode
        method to it, using eager execution.
        """
        # Load the Portuguese to English translation dataset
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)

        # Initialize tokenizers
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

        # Tokenize the dataset splits NOTE see tf.data.Dataset.map()
        self.data_train = self.data_train.map(
            self.tf_encode, num_parallel_calls=tf.data.AUTOTUNE)
        self.data_valid = self.data_valid.map(
            self.tf_encode, num_parallel_calls=tf.data.AUTOTUNE)

        # Set max_len attribute for convenience
        self.max_len = max_len

        # Filter, cache, shuffle, split into padded batches & prefetch data
        self.data_train = self.data_train.filter(self.filter_max_len)
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(buffer_size=20000)
        self.data_train = self.data_train.padded_batch(
            batch_size=batch_size,
            padded_shapes=([None], [None]))
        self.data_train = self.data_train.prefetch(tf.data.AUTOTUNE)

        # Only filter and split into padded batches for the validation data
        self.data_valid = self.data_valid.filter(self.filter_max_len)
        self.data_valid = self.data_valid.padded_batch(
            batch_size=batch_size,
            padded_shapes=([None], [None]))

    def filter_max_len(self, sentence_1, sentence_2):
        """
        Basic function to filter any example with either sentence of
        length higher than self.max_len tokens.

        :param sentence_1: sentence 1 (as: pt)
        :param sentence_2: sentence 2 (as: en)

        Returns:
        Boolean tensor (True if both sentences are under the
        max length, False otherwise).
        """
        return tf.logical_and(tf.size(sentence_1) <= self.max_len,
                              tf.size(sentence_2) <= self.max_len)

    def tokenize_dataset(self, data):
        """
        Tokenizes the dataset using pre-trained tokenizers and adapts them to
        the dataset.

        :param data: tf.data.Dataset containing tuples of (pt, en) sentences.

        Returns:
        - :tokenizer_pt: Trained tokenizer for Portuguese.
        - :tokenizer_en: Trained tokenizer for English.
        """
        # Get and decode sentences from the dataset (build iterator)
        pt_sentences = []
        en_sentences = []
        for pt, en in data.as_numpy_iterator():
            pt_sentences.append(pt.decode('utf-8'))
            en_sentences.append(en.decode('utf-8'))

        # Load the pre-trained tokenizers
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased', use_fast=True,
            clean_up_tokenization_spaces=True)
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased', use_fast=True,
            clean_up_tokenization_spaces=True)

        # Train both tokenizers on the dataset sentence iterators
        tokenizer_pt = tokenizer_pt.train_new_from_iterator(pt_sentences,
                                                            vocab_size=2**13)
        tokenizer_en = tokenizer_en.train_new_from_iterator(en_sentences,
                                                            vocab_size=2**13)

        # Update the Dataset tokenizers with the newly trained ones
        self.tokenizer_pt = tokenizer_pt
        self.tokenizer_en = tokenizer_en

        return self.tokenizer_pt, self.tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation into tokens, including start and end of sentence
        tokens.

        Args:
            pt: `tf.Tensor` containing the Portuguese sentence.
            en: `tf.Tensor` containing the corresponding English sentence.

        :returns pt_tokens, en_tokens:
        - :pt_tokens: `np.ndarray` containing the Portuguese tokens.
        - :en_tokens: `np.ndarray` containing the English tokens.
        """
        # Decode tf.Tensor to strings
        pt_sentence = pt.numpy().decode('utf-8')
        en_sentence = en.numpy().decode('utf-8')

        # Get the vocab_size from the tokenizers
        vocab_size_pt = self.tokenizer_pt.vocab_size
        vocab_size_en = self.tokenizer_en.vocab_size

        # Tokenize sentences with no special tokens
        pt_tokens = self.tokenizer_pt.encode(pt_sentence,
                                             add_special_tokens=False)
        en_tokens = self.tokenizer_en.encode(en_sentence,
                                             add_special_tokens=False)

        # Insert sentence start and end tokens
        pt_tokens = [vocab_size_pt] + pt_tokens + [vocab_size_pt + 1]
        en_tokens = [vocab_size_en] + en_tokens + [vocab_size_en + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        Tensorflow wrapper for the encode instance method
        Ensures the python method is ran eagerly.

        Returns the encoded token tensors.
        """
        pt_tokens, en_tokens = tf.py_function(func=self.encode,
                                              inp=[pt, en],
                                              Tout=[tf.int64, tf.int64])

        # Set the shapes of the tensors for the map method, just in case
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens
