#!/usr/bin/env python3
"""
Neural Style Transfer
"""


import numpy as np
import tensorflow as tf


class NST:
    """
    The NST class performs tasks for neural style transfer.

    Public Class Attributes:
    - style_layers: A list of layers to be used for style extraction,
    defaulting to ['block1_conv1', 'block2_conv1', 'block3_conv1',
    'block4_conv1', 'block5_conv1'].
    - content_layer: The layer to be used for content extraction,
    defaulting to 'block5_conv2'.
    """
    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initializes an NST instance.

        Parameters:
        - style_image (numpy.ndarray): The image used as a style reference.
        - content_image (numpy.ndarray): The image used as a content reference
        - alpha (float): The weight for content cost. Default is 1e4.
        - beta (float): The weight for style cost. Default is 1.

        Raises:
        - TypeError: If style_image is not a numpy.ndarray with
            shape (h, w, 3), raises an error with the message "style_image
            must be a numpy.ndarray with shape (h, w, 3)".
        - TypeError: If content_image is not a numpy.ndarray with
            shape (h, w, 3), raises an error with the message "content_image
            must be a numpy.ndarray with shape (h, w, 3)".
        - TypeError: If alpha is not a non-negative number, raises an error
            with the message "alpha must be a non-negative number".
        - TypeError: If beta is not a non-negative number, raises an error
            with the message "beta must be a non-negative number".

        Instance Attributes:
        - style_image: The preprocessed style image.
        - content_image: The preprocessed content image.
        - alpha: The weight for content cost.
        - beta: The weight for style cost.
        """
        if (not isinstance(style_image, np.ndarray)
                or style_image.shape[-1] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if (not isinstance(content_image, np.ndarray)
                or content_image.shape[-1] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")

        if not isinstance(alpha, (float, int)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (float, int)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixel values are between 0 and 1
        and its largest side is 512 pixels.

        Parameters:
        - image (numpy.ndarray): A numpy.ndarray of shape (h, w, 3) containing
        the image to be scaled.

        Raises:
        - TypeError: If image is not a numpy.ndarray with shape (h, w, 3),
          raises an error with the message "image must be a numpy.ndarray
          with shape (h, w, 3)".

        Returns:
        - tf.Tensor: The scaled image as a tf.Tensor with shape
          (1, h_new, w_new, 3), where max(h_new, w_new) == 512 and
          min(h_new, w_new) is scaled proportionately.
          The image is resized using bicubic interpolation, and its pixel
          values are rescaled from the range [0, 255] to [0, 1].
        """
        if (not isinstance(image, np.ndarray) or image.shape[-1] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w = image.shape[:2]

        if w > h:
            new_w = 512
            new_h = int((h * 512) / w)
        else:
            new_h = 512
            new_w = int((w * 512) / h)

        # Resize image (with bicubic interpolation)
        image_resized = tf.image.resize(
            image, size=[new_h, new_w],
            method=tf.image.ResizeMethod.BICUBIC)

        # Normalize pixel values to the range [0, 1]
        image_normalized = image_resized / 255

        # Clip values to ensure they are within [0, 1] range
        image_clipped = tf.clip_by_value(image_normalized, 0, 1)

        # Add batch dimension on axis 0 and return
        return tf.expand_dims(image_clipped, axis=0)
