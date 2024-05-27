#!/usr/bin/env python3
"""
Valid Convolution (grayscale)
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    Parameters:
    - images: numpy.ndarray with shape (m, h, w) containing multiple
    grayscale images
        - m: the number of images
        - h: the height in pixels of the images
        - w: the width in pixels of the images
    - kernel: numpy.ndarray with shape (kh, kw) containing the kernel
    for the convolution
        - kh: the height of the kernel
        - kw: the width of the kernel

    Returns:
    - A numpy.ndarray containing the convolved images
    """
    # Setup matrix dimensions
    m, h, w = images.shape
    kh, kw = kernel.shape
    # NOTE here the stride length is 1, so no dividing (n - f) by s
    output_h = h - kh + 1
    output_w = w - kw + 1

    # Initialize the output array
    convolved = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            # slice region from each image, using kernel shape
            region = images[:, i:(i + kh), j:(j + kw)]

            # Convolve each image (m) for this region (i, j)
            convolved[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return convolved
