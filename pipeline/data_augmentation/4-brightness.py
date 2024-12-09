#!/usr/bin/env python3
"""
Randomly adjust the brightness of an image with TensorFlow
"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly adjusts the brightness of an image.

    Args:
        - image (tf.Tensor): the input image to adjust.
        - max_delta (non-negative float): maximum relative change in brightness
    Returns:
        tf.Tensor: the altered image
    """
    return tf.image.random_brightness(image=image, max_delta=max_delta)
