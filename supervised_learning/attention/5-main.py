#!/usr/bin/env python3

import os
import random
import numpy as np
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention

SEED = 0

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

Q = tf.convert_to_tensor(np.random.uniform(
    size=(50, 10, 256)).astype('float32'))
K = tf.convert_to_tensor(np.random.uniform(
    size=(50, 15, 256)).astype('float32'))
V = tf.convert_to_tensor(np.random.uniform(
    size=(50, 15, 512)).astype('float32'))
output, weights = sdp_attention(Q, K, V)
print(output)
print(weights)
