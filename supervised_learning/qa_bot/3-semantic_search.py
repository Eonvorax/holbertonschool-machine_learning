#!/usr/bin/env python3
"""
Semantic search
"""

import os
from sentence_transformers import SentenceTransformer
import numpy as np


def cosine_similarity(vec1, vec2):
    """
    Compute the cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity value
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def semantic_search(corpus_path, sentence):
    """
    Perform semantic search on a corpus of documents.

    Args:
        corpus_path: Path to the corpus of reference documents
        sentence: The sentence to search with

    Returns:
        The reference text of the document most similar to the input sentence
    """
    # Load a pre-trained Sentence-BERT model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Step 1: Read the corpus documents
    documents = []
    file_names = os.listdir(corpus_path)
    for file_name in file_names:
        if file_name.endswith('.md'):
            file_path = os.path.join(corpus_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                documents.append(f.read())

    # Generate embeddings for the corpus documents
    doc_embeddings = model.encode(documents)

    # Generate an embedding for the input sentence
    query_embedding = model.encode([sentence])[0]

    # Compute cosine similarities between the query and each document
    similarities = [cosine_similarity(query_embedding, doc_embedding)
                    for doc_embedding in doc_embeddings]

    # Find index of document with highest similarity score
    best_doc_index = np.argmax(similarities)

    # Return the most similar document
    return documents[best_doc_index]
