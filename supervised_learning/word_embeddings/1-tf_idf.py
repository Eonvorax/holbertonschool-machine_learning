#!/usr/bin/env python3
"""
Bag of words embedding matrix
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix.

    :param sentences:
        A list of sentences to analyze

    :param vocab:
        sentences: list of sentences to analyze
        vocab: list of vocabulary words to use for the analysis (optional).
        If None, all words within sentences should be used

    Returns:
    `(embeddings, features)`
    - `embeddings`: numpy.ndarray of shape `(s, f)` containing the embeddings
      - `s` is the number of sentences
      - `f` is the number of features analyzed
    - `features`: list of the features used for embeddings
    """
    # Initialize the  TF-IDF vectorizer with the given vocabulary
    vectorizer = TfidfVectorizer(vocabulary=vocab)

    # Fit + transform the sentences to get the TF-IDF embeddings
    embeddings = vectorizer.fit_transform(sentences)

    # Extract the features (words) used by the vectorizer
    features = vectorizer.get_feature_names_out()

    return embeddings.toarray(), features
