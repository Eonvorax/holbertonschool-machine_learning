#!/usr/bin/env python3

import tensorflow as tf


def flip_image(image):
    """
    Flips a tensor image horizontally

    Args:
        image (tf.Tensor): the given image to flip

    Returns:
        (tf.Tensor): the flipped image
    """
    return tf.image.flip_left_right(image)
