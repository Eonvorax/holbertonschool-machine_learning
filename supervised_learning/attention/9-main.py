#!/usr/bin/env python3

import os
import random
import numpy as np
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder

SEED = 0

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

encoder = Encoder(6, 512, 8, 2048, 10000, 1000)
print(encoder.dm)
print(encoder.N)
print(encoder.embedding)
print(encoder.positional_encoding)
print(encoder.blocks)
print(encoder.dropout)
x = tf.random.uniform((32, 10))
output = encoder(x, True, None)
print(output)
