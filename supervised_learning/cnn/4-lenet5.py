#!/usr/bin/env python3
"""
LeNet-5 (Tensorflow 1)
"""

import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
    Builds a modified version of the LeNet-5 architecture using tensorflow.

    Args:
    x (tf.placeholder): Placeholder for the input images for the network.
                        It should have the shape (m, 28, 28, 1) where m is
                        the number of images.
    y (tf.placeholder): Placeholder for the one-hot labels for the network.
                        It should have the shape (m, 10).

    The model consists of the following layers in order:
    - Convolutional layer with 6 kernels of shape 5x5 with same padding
    - Max pooling layer with kernels of shape 2x2 with 2x2 strides
    - Convolutional layer with 16 kernels of shape 5x5 with valid padding
    - Max pooling layer with kernels of shape 2x2 with 2x2 strides
    - Fully connected layer with 120 nodes
    - Fully connected layer with 84 nodes
    - Fully connected softmax output layer with 10 nodes

    All layers requiring initialization initialize their kernels with the
    he_normal initialization method:
        tf.keras.initializers.VarianceScaling(scale=2.0).
    All hidden layers requiring activation use the relu activation function.

    Returns:
    tuple: a tuple containing:
        - A tensor for the softmax activated output
        - A training operation that utilizes Adam optimization
        - A tensor for the loss of the network
        - A tensor for the accuracy of the network
    """
    init = tf.keras.initializers.VarianceScaling(scale=2.0)
    conv2d_1 = tf.layers.Conv2D(
        filters=6,
        kernel_size=5,
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(x)

    max_pooling_1 = tf.layers.MaxPooling2D(
        pool_size=2,
        strides=2
    )(conv2d_1)

    conv2d_2 = tf.layers.Conv2D(
        filters=16,
        kernel_size=5,
        padding='valid',
        activation='relu',
        kernel_initializer=init
    )(max_pooling_1)

    max_pooling_2 = tf.layers.MaxPooling2D(
        pool_size=2,
        strides=2
    )(conv2d_2)

    # Flatten tensor to 1D tensor, to match Dense layer dimensions
    flattened = tf.layers.Flatten()(max_pooling_2)

    fc1 = tf.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=init
    )(flattened)

    fc2 = tf.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=init
    )(fc1)

    output = tf.layers.Dense(
        units=10,
        kernel_initializer=init
    )(fc2)
    # NOTE no built-in activation function (need non-softmaxed output)

    softmax = tf.nn.softmax(output)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=output)
    train_op = tf.train.AdamOptimizer().minimize(loss)

    # Compare prediction to true labels
    y_pred = tf.argmax(output, axis=1)
    y_true = tf.argmax(y, axis=1)
    correct_prediction = tf.equal(y_pred, y_true)

    # Accuracy: average success rate (converted booleans to float32)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

    return softmax, train_op, loss, accuracy
