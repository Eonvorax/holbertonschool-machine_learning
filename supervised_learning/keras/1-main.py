#!/usr/bin/env python3

# Force Seed - fix for Keras
import tensorflow as tf
import numpy as np
import random
import os
SEED = 8

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

build_model = __import__('1-input').build_model

if __name__ == '__main__':
    network = build_model(784, [256, 256, 10], [
                          'tanh', 'tanh', 'softmax'], 0.001, 0.95)
    network.summary()
    print(network.losses)
