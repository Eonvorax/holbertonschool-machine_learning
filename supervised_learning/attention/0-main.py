#!/usr/bin/env python3

import os
import random
import numpy as np
import tensorflow as tf
RNNEncoder = __import__('0-rnn_encoder').RNNEncoder

SEED = 0

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

encoder = RNNEncoder(1024, 128, 256, 32)
print(encoder.batch)
print(encoder.units)
print(type(encoder.embedding))
print(type(encoder.gru))

initial = encoder.initialize_hidden_state()
print(initial)
x = tf.convert_to_tensor(np.random.choice(1024, 320).reshape((32, 10)))
outputs, hidden = encoder(x, initial)
print(outputs)
print(hidden)
