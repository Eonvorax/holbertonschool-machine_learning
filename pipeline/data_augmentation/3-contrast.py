#!/usr/bin/env python3
"""
Randomly adjust the contrast of an image with TensorFlow
"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image.

    Args:
        - image (tf.Tensor): the input image to adjust.
        - lower (float): the lower bound of the random contrast factor range.
        - upper (float): the upper bound of the random contrast factor range.

    Returns:
        tf.Tensor: the contrast-adjusted image
    """
    return tf.image.random_contrast(image=image, lower=lower, upper=upper)
