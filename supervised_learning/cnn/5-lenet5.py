#!/usr/bin/env python3
"""
LeNet-5 (Keras)
"""
from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified LeNet-5 architecture using Keras.

    Args:
    X: K.Input of shape (m, 28, 28, 1) containing the input images.
        m is the number of images.

    The model consists of the following layers in order:
    - Convolutional layer with 6 kernels of shape 5x5 with same padding
    - Max pooling layer with kernels of shape 2x2 with 2x2 strides
    - Convolutional layer with 16 kernels of shape 5x5 with valid padding
    - Max pooling layer with kernels of shape 2x2 with 2x2 strides
    - Fully connected layer with 120 nodes
    - Fully connected layer with 84 nodes
    - Fully connected softmax output layer with 10 nodes

    Returns:
        A K.Model compiled to use Adam optimization
        (with default hyperparameters) and accuracy metrics.
    """

    initializer = K.initializers.HeNormal(seed=0)
    model = K.Sequential()

    model.add(X)

    model.add(K.layers.Conv2D(filters=6,
                              kernel_size=5,
                              padding='same',
                              kernel_initializer=initializer,
                              activation='relu'))

    model.add(K.layers.MaxPooling2D(pool_size=2, strides=2))

    model.add(K.layers.Conv2D(filters=16,
                              kernel_size=5,
                              padding='valid',
                              kernel_initializer=initializer,
                              activation='relu'))

    model.add(K.layers.MaxPooling2D(pool_size=2, strides=2))

    model.add(K.layers.Flatten())

    model.add(K.layers.Dense(units=120,
                             kernel_initializer=initializer,
                             activation='relu'))

    model.add(K.layers.Dense(units=84,
                             kernel_initializer=initializer,
                             activation='relu'))

    model.add(K.layers.Dense(units=10,
                             kernel_initializer=initializer,
                             activation='softmax'))

    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
