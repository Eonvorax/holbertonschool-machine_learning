#!/usr/bin/env python3
"""
Builds, trains, and saves a neural network classifier.
"""

import tensorflow.compat.v1 as tf

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier.

    Arguments:
    X_train: numpy.ndarray containing the training input data
    Y_train: numpy.ndarray containing the training labels
    X_valid: numpy.ndarray containing the validation input data
    Y_valid: numpy.ndarray containing the validation labels
    layer_sizes: list containing the number of nodes in each layer
    activations: list containing the activation functions for each layer
    alpha: the learning rate
    iterations: the number of iterations to train over
    save_path: designates where to save the model

    Returns:
    the path where the model was saved
    """
    nx = X_train.shape[1]
    classes = Y_train.shape[1]

    # Use previous functions to prepare placeholders & tensors
    x, y = create_placeholders(nx, classes)
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)

    # Add an op to initialize the variables.
    init_op = tf.global_variables_initializer()

    # Add ops to save and restore all the variables (saver object)
    saver = tf.train.Saver()

    # Add tensors to graph's collection
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    with tf.Session() as sess:
        # Initialize variables
        sess.run(init_op)

        # Training loop
        for i in range(iterations + 1):
            train_loss, train_accuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_train, y: Y_train})
            valid_loss, valid_accuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

            # Print metrics after every 100 iterations, the 0th & the last
            if i % 100 == 0 or i == iterations:
                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {train_loss}")
                print(f"\tTraining Accuracy: {train_accuracy}")
                print(f"\tValidation Cost: {valid_loss}")
                print(f"\tValidation Accuracy: {valid_accuracy}")

            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})

        # Save the model
        save_path = saver.save(sess, save_path)

    return save_path
