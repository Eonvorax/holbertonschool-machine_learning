#!/usr/bin/env python3
"""
Evaluates the output of a neural network.
"""

import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """
    Evaluates the output of a neural network.

    Args:
        X (ndarray): the input data to evaluate.
        Y (ndarray): the one-hot labels for X.
        save_path (str): the file to load the model from.

    Returns:
        The network's prediction, accuracy, and loss.
    """

    with tf.Session() as sess:
        # Restore model from file
        saver = tf.train.import_meta_graph(save_path + ".meta")
        saver.restore(sess, save_path)

        # Get placeholders and tensors from collection
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]

        # Compute prediction, accuracy and loss
        prediction = sess.run(y_pred, feed_dict={x: X, y: Y})
        accuracy = sess.run(accuracy, feed_dict={x: X, y: Y})
        loss = sess.run(loss, feed_dict={x: X, y: Y})

        return prediction, accuracy, loss
