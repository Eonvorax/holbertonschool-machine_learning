#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    Question Answer
    """
    tokenizer = BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad")
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

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
    start_index = tf.math.argmax(
        start_logits[0, 1:sequence_length-1], output_type=tf.int32).numpy() + 1
    end_index = tf.math.argmax(
        end_logits[0, 1:sequence_length-1], output_type=tf.int32).numpy() + 1

    # Get the answer tokens
    answer_tokens = inputs["input_ids"][0][start_index: end_index + 1].numpy()

    # Decode the answer tokens
    answer = tokenizer.decode(
        answer_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    if not answer.strip():
        return None

    return answer


with open('ZendeskArticles/PeerLearningDays.md') as f:
    reference = f.read()

print(question_answer('When are PLDs?', reference))
