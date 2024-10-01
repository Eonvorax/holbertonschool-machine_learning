#!/usr/bin/env python3

import tensorflow as tf
Dataset = __import__('3-dataset').Dataset

tf.random.set_seed(0)
data = Dataset(32, 40)
for pt, en in data.data_train.take(1):
    print(pt, en)
for pt, en in data.data_valid.take(1):
    print(pt, en)
