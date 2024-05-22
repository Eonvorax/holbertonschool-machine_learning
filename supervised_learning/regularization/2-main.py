#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import os
import random

SEED = 0

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

l2_reg_cost = __import__('2-l2_reg_cost').l2_reg_cost


def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    m = Y.shape[0]
    oh = np.zeros((m, classes))
    oh[np.arange(m), Y] = 1
    return oh


m = np.random.randint(1000, 2000)
c = 10
lib = np.load('MNIST.npz')

X = lib['X_train'][:m].reshape((m, -1))
Y = one_hot(lib['Y_train'][:m], c)

model_reg = tf.keras.models.load_model('model_reg.h5', compile=False)

Predictions = model_reg(X)
cost = tf.keras.losses.CategoricalCrossentropy()(Y, Predictions)

l2_cost = l2_reg_cost(cost, model_reg)
print(l2_cost)
