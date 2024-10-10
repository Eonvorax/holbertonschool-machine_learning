#!/usr/bin/env python3
"""
Multi-reference Question Answering
"""

import os
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np

tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad")
model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
sbert = SentenceTransformer('all-MiniLM-L6-v2')


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

    # Read the corpus documents
    documents = []
    file_names = os.listdir(corpus_path)
    for file_name in file_names:
        if file_name.endswith('.md'):
            file_path = os.path.join(corpus_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                documents.append(f.read())

    # Generate embeddings for the corpus documents
    doc_embeddings = sbert.encode(documents)

    # Generate an embedding for the input sentence
    query_embedding = sbert.encode([sentence])[0]

    # Compute cosine similarities between the query and each document
    similarities = [cosine_similarity(query_embedding, doc_embedding)
                    for doc_embedding in doc_embeddings]

    # Find index of document with highest similarity score
    best_doc_index = np.argmax(similarities)

    # Return the most similar document
    return documents[best_doc_index]


def find_answer(question, reference):
    """
    Finds a snippet of text within a reference document to answer a question.
    - Uses the `bert-uncased-tf2-qa` model from the `tensorflow-hub` library
    - Uses the pre-trained `BertTokenizer` from the `transformers` library,
    `bert-large-uncased-whole-word-masking-finetuned-squad`.

    Args:
        question: A string containing the question to answer
        reference: A string containing the reference document from which to
            find the answer

    Returns:
        A string containing the answer. If no answer is found, return `None`
    """
    # Tokenize the input using the BERT tokenizer
    inputs = tokenizer(question, reference, return_tensors="tf")

    # Prepare the input for the TensorFlow Hub model
    input_tensors = [
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["token_type_ids"]
    ]

    # Pass the input tensors to the model
    output = model(input_tensors)

    # Access the start and end logits
    start_logits = output[0]
    end_logits = output[1]

    # Get the input sequence length
    sequence_length = inputs["input_ids"].shape[1]

    # Find the best start and end indices within the input sequence
    start_index = tf.math.argmax(start_logits[0, 1:sequence_length - 1]) + 1
    end_index = tf.math.argmax(end_logits[0, 1:sequence_length - 1]) + 1

    # Get the answer tokens using the best indices
    answer_tokens = inputs["input_ids"][0][start_index: end_index + 1]

    # Decode the answer tokens
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True,
                              clean_up_tokenization_spaces=True)

    # No answer ? Returning None
    if not answer.strip():
        return None

    return answer


def question_answer(corpus_path):
    """
    Continuously prompts the user for input until one of the predefined
    farewell words is entered. Answers questions from a given corpus of
    markdown documents using a BERT tokenizer and model.

    Args:
        corpus_path: the path to the directory containing the documents.
    """

    farewells = ["exit", "quit", "goodbye", "bye"]

    user_input = ""
    while True:
        user_input = input("Q: ")

        # Check user input for farewell string (case insensitive)
        if user_input.lower() in farewells:
            print("A: Goodbye")
            return

        # Response to any other input:
        # Semantic search using user input to locate the best document
        best_document = semantic_search(corpus_path, user_input)

        # Use the best document to try and find an answer substring in it
        answer = find_answer(question=user_input, reference=best_document)
        if answer is None:
            print("A: Sorry, I do not understand your question.")
        else:
            print(f"A: {answer}")
