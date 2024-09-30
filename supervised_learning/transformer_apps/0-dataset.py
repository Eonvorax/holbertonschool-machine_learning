#!/usr/bin/env python3
"""
Load, tokenize tensorflow Dataset
"""

import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    A class to load and prepare the TED HRLR translation dataset
    for machine translation from Portuguese to English.
    """

    def __init__(self):
        """
        Initializes the Dataset object and loads the training and validation
        datasets.
        Also initializes tokenizers for Portuguese and English.
        """
        # Load the Portuguese to English translation dataset
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)

        # Initialize tokenizers
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

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
