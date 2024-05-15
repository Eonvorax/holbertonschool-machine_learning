#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import os

SEED = 0

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

learning_rate_decay = __import__('12-learning_rate_decay').learning_rate_decay


def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    one_hot = np.zeros((Y.shape[0], classes))
    one_hot[np.arange(Y.shape[0]), Y] = 1
    return one_hot


lib = np.load('MNIST.npz')
X_3D = lib['X_train']
Y = lib['Y_train']
X = X_3D.reshape((X_3D.shape[0], -1))
Y_oh = one_hot(Y, 10)

model = tf.keras.models.load_model('model.h5', compile=False)

alpha = 0.1
alpha_schedule = learning_rate_decay(alpha, 1, 10)
optimizer = tf.keras.optimizers.SGD(learning_rate=alpha_schedule)


@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = tf.reduce_mean(
            tf.keras.losses.CategoricalCrossentropy()(labels, predictions))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


total_iterations = 100
for iteration in range(total_iterations):

    current_learning_rate = alpha_schedule(iteration).numpy()
    print(current_learning_rate)
    cost = train_step(X, Y_oh)
