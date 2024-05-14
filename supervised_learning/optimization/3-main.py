#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import random
import os

SEED = 0

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS']= '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

create_mini_batches = __import__('3-mini_batch').create_mini_batches


def one_hot(Y, classes):
    """Convert an array to a one-hot matrix"""
    one_hot = np.zeros((Y.shape[0], classes))
    one_hot[np.arange(Y.shape[0]), Y] = 1
    return one_hot

lib = np.load('MNIST.npz')
X_3D = lib['X_train']
Y = lib['Y_train']
X = X_3D.reshape((X_3D.shape[0], -1))
Y_oh = one_hot(Y, 10)
X_valid_3D = lib['X_valid']
Y_valid = lib['Y_valid']
X_valid = X_valid_3D.reshape((X_valid_3D.shape[0], -1))
Y_valid_oh = one_hot(Y_valid, 10)

model = tf.keras.models.load_model('model.h5', compile=False)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

batch_size = 32
epochs = 10

loss_fn = tf.keras.losses.CategoricalCrossentropy()

for epoch in range(epochs):
    print(f"After {epoch} epochs:")

    train_loss = tf.reduce_mean(loss_fn(Y_oh, model(X)))
    train_accuracy = np.mean(np.argmax(model(X), axis=1) == Y)

    valid_loss = tf.reduce_mean(loss_fn(Y_valid_oh, model(X_valid)))
    valid_accuracy = np.mean(np.argmax(model(X_valid), axis=1) == Y_valid)

    print(f"\tTraining Cost: {train_loss}")
    print(f"\tTraining Accuracy: {train_accuracy}")
    print(f"\tValidation Cost: {valid_loss}")
    print(f"\tValidation Accuracy: {valid_accuracy}")

    for step, (X_batch, Y_batch) in enumerate(create_mini_batches(X, Y_oh, batch_size)):
        with tf.GradientTape() as tape:
            predictions = model(X_batch)
            loss = loss_fn(Y_batch, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if (step + 1) % 100 == 0:
            Y_pred = np.argmax(predictions, axis=1)
            batch_accuracy = np.mean(Y_pred == np.argmax(Y_batch, axis=1))
            print(f"\tStep {step + 1}:")
            print(f"\t\tCost: {loss}")
            print(f"\t\tAccuracy: {batch_accuracy}")

print(f"After {epochs} epochs:")

final_train_loss = tf.reduce_mean(loss_fn(Y_oh, model(X)))
final_train_accuracy = np.mean(np.argmax(model(X), axis=1) == Y)

final_valid_loss = tf.reduce_mean(loss_fn(Y_valid_oh, model(X_valid)))
final_valid_accuracy = np.mean(np.argmax(model(X_valid), axis=1) == Y_valid)

print(f"\tFinal Training Cost: {final_train_loss}, Accuracy: {final_train_accuracy}")
print(f"\tFinal Validation Cost: {final_valid_loss}, Accuracy: {final_valid_accuracy}")
