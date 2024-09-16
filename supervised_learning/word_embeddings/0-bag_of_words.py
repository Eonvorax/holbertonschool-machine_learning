#!/usr/bin/env python3
"""
Bag of words embedding matrix
"""

import string
import re
import numpy as np


def preprocess_sentences(sentences):
    """
    Preprocess the given list of sentences, normalizing all characters to
    lowercase, removing possessive `'s`, and removing punctuation.
    """
    preprocessed_sentences = []

    for sentence in sentences:
        # Normalize the sentence to lowercase
        processed_sentence = sentence.lower()

        # Remove possessive "'s" (for example: "children's" -> "children")
        processed_sentence = re.sub(r"\'s\b", "", processed_sentence)

        # Remove remaining apostrophes and other punctuation
        processed_sentence = re.sub(
            f"[{re.escape(string.punctuation)}]", "", processed_sentence)

        # Split the sentence into words and append to the list
        preprocessed_sentences.append(processed_sentence.split())

    return preprocessed_sentences


def build_vocab(processed_sentences):
    """
    Build the vocabulary from the given preprocessed list of sentences
    """
    return sorted(set(
        word for sentence in processed_sentences for word in sentence))


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix:

    :param sentences:
        A list of sentences to analyze

    :param vocab:
        A list of the vocabulary words to use for the analysis.
        If `None`, all words within sentences should be used.
        Defaults to `None`.

    Returns:
    `(embeddings, features)`
    - `embeddings` is a numpy.ndarray of shape `(s, f)` containing embeddings
        - `s` is the number of sentences in sentences
        - `f` is the number of features analyzed
    - `features` is a list of the features used for `embeddings`
    """
    # Preprocess each sentence
    processed_sentences = preprocess_sentences(sentences)

    # If vocab is not provided, build the vocabulary from the sentences
    if vocab is None:
        vocab = build_vocab(processed_sentences)

    # Mapping from each word to its index in the vocab
    word_to_index = {word: i for i, word in enumerate(vocab)}

    # Initialize an embedding matrix of zeros with shape (s, f)
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)
    features = np.array(vocab)

    # Fill the embedding matrix
    for i, sentence in enumerate(processed_sentences):
        for word in sentence:
            if word in word_to_index:
                embeddings[i, word_to_index[word]] += 1

    return embeddings, features
