#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import random
import os

SEED = 0

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

create_batch_norm_layer = __import__('14-batch_norm').create_batch_norm_layer

lib = np.load('MNIST.npz')
X_3D = lib['X_train']
X = X_3D.reshape((X_3D.shape[0], -1))

a = create_batch_norm_layer(X, 256, tf.nn.tanh)
print(a)
