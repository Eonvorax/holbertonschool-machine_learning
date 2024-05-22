#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import random
import os

SEED = 4

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

dropout_create_layer = __import__(
    '6-dropout_create_layer').dropout_create_layer

X = np.random.randint(0, 256, size=(10, 784))
a = dropout_create_layer(X, 256, tf.nn.tanh, 0.8)
print(a[0])
