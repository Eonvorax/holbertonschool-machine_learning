#!/usr/bin/env python3
"""
Crop an image randomly with TensorFlow
"""
import tensorflow as tf


def crop_image(image, size):
    """
    Performs a random crop of an image.

    Args:
        - image (tf.Tensor): a 3D tf.Tensor containing the image to crop
        - size (tuple of ints): the dimensions of the crop

    Returns:
        tf.Tensor: the resulting randomly cropped image
    """
    return tf.image.random_crop(image, size=size)
