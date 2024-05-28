#!/usr/bin/env python3
"""
Convolution with Padding
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.

    Parameters:
    - images: numpy.ndarray with shape (m, h, w) containing multiple
    grayscale images
    - kernel: numpy.ndarray with shape (kh, kw) containing the kernel
    for the convolution
    - padding: tuple of (ph, pw)
        - ph: the padding for the height of the image
        - pw: the padding for the width of the image

    Returns:
    - A numpy.ndarray containing the convolved images
    """
    # Setup matrixes and padding dimensions
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Calculate output dimensions (accounting for padding this time)
    output_h = h + 2 * ph - kh + 1
    output_w = w + 2 * pw - kw + 1

    # NOTE padding indexes: (before, after), shortcut is (padding,)
    padded_imgs = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    convolved = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            # Extract region from padded images
            region = padded_imgs[:, i:i+kh, j:j+kw]

            # Convolve each image (m) for this region (i, j)
            convolved[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return convolved
