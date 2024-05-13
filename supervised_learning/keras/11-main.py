#!/usr/bin/env python3
"""
Main file
"""

# Force Seed - fix for Keras
SEED = 8

import os
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS']= '0'
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)


# Imports
model = __import__('9-model')
config = __import__('11-config')

if __name__ == '__main__':
    network = model.load_model('network1.keras')
    config.save_config(network, 'config1.json')
    del network

    network2 = config.load_config('config1.json')
    network2.summary()
    print(network2.get_weights())
