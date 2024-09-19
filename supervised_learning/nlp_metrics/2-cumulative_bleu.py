#!/usr/bin/env python3
"""
Cumulative N-gram BLEU score
"""

import numpy as np
from collections import Counter


def get_ngrams(sentence, n):
    """
    Extracts n-grams from a sentence.

    :param sentence: list of words in the sentence
    :param n: the size of the n-gram
    :return: list of n-grams (as tuples)
    """
    return [tuple(sentence[i:i+n]) for i in range(len(sentence) - n + 1)]


def count_matches(references, sentence_ngrams, n):
    """
    Counts the number of matching n-grams between the candidate sentence and
    the references.

    :param references: list of reference translations
    :param sentence_ngrams: n-grams of the candidate sentence
    :param n: the size of the n-gram
    :return: number of matching n-grams
    """
    sentence_ngrams_count = Counter(sentence_ngrams)

    max_ref_ngrams = Counter()
    for reference in references:
        ref_ngrams = get_ngrams(reference, n)
        ref_ngrams_count = Counter(ref_ngrams)
        # Keep the maximum count for each n-gram across references
        for ngram in ref_ngrams_count:
            max_ref_ngrams[ngram] = max(
                max_ref_ngrams[ngram], ref_ngrams_count[ngram])

    # matches: n-grams in sentence that appear in any reference, for max counts
    matches = 0
    for ngram in sentence_ngrams_count:
        matches += min(sentence_ngrams_count[ngram],
                       max_ref_ngrams.get(ngram, 0))

    return matches


def calculate_brevity_penalty(references, sentence_len):
    """
    Calculates brevity penalty based on sentence length and reference length.

    :param references: list of reference translations
    :param sentence_len: length of the candidate sentence
    :return: brevity penalty (float)
    """
    # Find the reference length closest to the sentence length
    closest_ref_len = min((abs(len(ref) - sentence_len), len(ref))
                          for ref in references)[1]

    # Calc. brevity penalty
    if sentence_len > closest_ref_len:
        return 1
    else:
        return np.exp(1 - closest_ref_len / sentence_len)


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence.

    :param references: list of reference translations, each translation
    is a list of the words
    :param sentence: list containing the model proposed sentence
    :param n: the size of the largest n-gram to use for evaluation
    :return: the cumulative n-gram BLEU score
    """
    precisions = []

    # Calculate BLEU scores for all n-grams from 1 to n
    for i in range(1, n + 1):
        sentence_ngrams = get_ngrams(sentence, i)
        matches = count_matches(references, sentence_ngrams, i)

        # Calculate precision for current n-gram
        total_ngrams = len(sentence_ngrams)
        precision = matches / total_ngrams if total_ngrams > 0 else 0
        precisions.append(precision)

    # Calculate geometric mean of precisions
    if any(precision == 0 for precision in precisions):
        geometric_mean_precision = 0
    else:
        geometric_mean_precision = np.exp(np.sum(np.log(precisions)) / n)

    # Calculate brevity penalty
    brevity_penalty = calculate_brevity_penalty(references, len(sentence))

    # Return final cumulative BLEU score
    return brevity_penalty * geometric_mean_precision
