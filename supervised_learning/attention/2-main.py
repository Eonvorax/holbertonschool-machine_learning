#!/usr/bin/env python3

import os
import random
import numpy as np
import tensorflow as tf
RNNDecoder = __import__('2-rnn_decoder').RNNDecoder

SEED = 0

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

decoder = RNNDecoder(2048, 128, 256, 32)
print(decoder.embedding)
print(decoder.gru)
print(decoder.F)
x = tf.convert_to_tensor(np.random.choice(2048, 32).reshape((32, 1)))
s_prev = tf.convert_to_tensor(
    np.random.uniform(size=(32, 256)).astype('float32'))
hidden_states = tf.convert_to_tensor(
    np.random.uniform(size=(32, 10, 256)).astype('float32'))
y, s = decoder(x, s_prev, hidden_states)
print(y)
print(s)
