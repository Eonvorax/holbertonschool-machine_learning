#!/usr/bin/env python3
"""
Word2Vec model training
"""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a Word2Vec model.

    :Parameters:
    - sentences: list of tokenized sentences to be trained on
    - vector_size: dimensionality of the embedding layer
    - min_count: minimum number of occurrences of a word for use in training
    - window: maximum distance between the current and predicted word within
    a sentence
    - negative: size of negative sampling
    - cbow: boolean to determine training type; True is for CBOW, False for
    Skip-gram
    - epochs: number of iterations (epochs) to train over
    - seed: seed for the random number generator
    - workers: number of worker threads to train the model

    Returns:
    - The trained Word2Vec model
    """
    if cbow:
        sg = 0
    else:
        sg = 1

    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        epochs=epochs,
        seed=seed,
        workers=workers
    )
    # Prepare the model's vocabulary and train it
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count,
                epochs=model.epochs)
    return model
