#!/usr/bin/env python3
"""
Rotate an image with TensorFlow
"""
import tensorflow as tf


def rotate_image(image):
    """
    Rotates an image by 90 degrees counter-clockwise.

    Args:
        image (tf.Tensor): a tf.Tensor containing the image to rotate

    Return:
        tf.Tensor: the resulting rotated image.
    """
    return tf.image.rot90(image=image, k=1)
