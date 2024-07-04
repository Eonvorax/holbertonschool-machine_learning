#!/usr/bin/env python3

import matplotlib.image as mpimg
import os
import random
import numpy as np
import tensorflow as tf

NST = __import__('4-neural_style').NST


if __name__ == '__main__':
    style_image = mpimg.imread("starry_night.jpg")
    content_image = mpimg.imread("golden_gate.jpg")

    nst = NST(style_image, content_image)

    # Reproducibility
    SEED = 0
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_ENABLE_ONEDNN_OPTS']= '0'
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    vgg19 = tf.keras.applications.vgg19
    preprocecced = vgg19.preprocess_input(nst.content_image * 255)
    outputs = nst.model(preprocecced)
    layer_style_cost = nst.layer_style_cost(outputs[0], nst.gram_style_features[0])
    print(layer_style_cost)