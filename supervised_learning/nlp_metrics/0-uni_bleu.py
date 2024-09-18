#!/usr/bin/env python3
"""
Unigram BLEU score
"""
import numpy as np


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence:

    :Parameters:
    - `references` is a list of reference translations
        - each reference translation is a list of the words in the translation
    - `sentence` is a list containing the model proposed sentence

    Returns: the unigram BLEU score
    """

    sentence_len = len(sentence)

    # Calculate maximum matches for each word in the sentence
    matches = 0
    for word in set(sentence):
        max_count = max(ref.count(word) for ref in references)
        matches += min(sentence.count(word), max_count)

    # Calculate precision (correct matches / total words in sentence)
    precision = matches / sentence_len

    # Find the reference length closest to the sentence length
    closest_ref_len = min(
        (abs(len(ref) - sentence_len), len(ref)) for ref in references)[1]

    # Calculate brevity penalty
    if sentence_len > closest_ref_len:
        brevity_penalty = 1
    else:
        brevity_penalty = np.exp(1 - closest_ref_len / sentence_len)

    # Final BLEU score, accounting for BP if necessary
    return brevity_penalty * precision
