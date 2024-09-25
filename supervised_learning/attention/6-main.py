#!/usr/bin/env python3

import os
import random
import numpy as np
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention

SEED = 0

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

mha = MultiHeadAttention(512, 8)
print(mha.dm)
print(mha.h)
print(mha.depth)
print(mha.Wq)
print(mha.Wk)
print(mha.Wv)
print(mha.linear)
Q = tf.convert_to_tensor(np.random.uniform(
    size=(50, 15, 256)).astype('float32'))
K = tf.convert_to_tensor(np.random.uniform(
    size=(50, 15, 256)).astype('float32'))
V = tf.convert_to_tensor(np.random.uniform(
    size=(50, 15, 256)).astype('float32'))
output, weights = mha(Q, K, V, None)
print(output)
print(weights)
