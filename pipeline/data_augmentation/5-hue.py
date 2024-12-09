#!/usr/bin/env python3
"""
Adjust the hue of an image with TensorFlow
"""
import tensorflow as tf


def change_hue(image, delta):
    """
    Changes the hue of an image.

    Args:
        - image (tf.Tensor): the input image to adjust.
        - delta (float): how much to add to the hue channel.

    Return:
        tf.Tensor: the altered image
    """
    return tf.image.adjust_hue(image=image, delta=delta)
