#!/usr/bin/env python3

import matplotlib.image as mpimg
import os
import random
import numpy as np
import tensorflow as tf

NST = __import__('2-neural_style').NST


if __name__ == '__main__':
    style_image = mpimg.imread("starry_night.jpg")
    content_image = mpimg.imread("golden_gate.jpg")

    # Reproducibility
    SEED = 0
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    nst = NST(style_image, content_image)
    input_layer = tf.constant(np.random.randn(1, 28, 30, 3), dtype=tf.float32)
    print(input_layer)
    gram_matrix = nst.gram_matrix(input_layer)
    print(gram_matrix)
