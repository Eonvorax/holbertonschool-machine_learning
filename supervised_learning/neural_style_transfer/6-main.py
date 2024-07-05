#!/usr/bin/env python3

import matplotlib.image as mpimg
import os
import random
import numpy as np
import tensorflow as tf

NST = __import__('6-neural_style').NST


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
    generated_image = np.random.uniform(size=nst.content_image.shape)
    generated_image = generated_image.astype('float32')
    vgg19 = tf.keras.applications.vgg19
    preprocecced = vgg19.preprocess_input(generated_image * 255)
    outputs = nst.model(preprocecced)
    content_output = outputs[-1]
    content_cost = nst.content_cost(content_output)
    print(content_cost)
