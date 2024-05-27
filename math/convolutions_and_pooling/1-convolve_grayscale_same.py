#!/usr/bin/env python3
"""
Same Convolution (grayscale)
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

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
    # NOTE same shape of output when using same convolution
    # output_h = h
    # output_w = w

    # Initialize the output array
    convolved = np.zeros((m, h, w))
    # NOTE padding indexes: (before, after), shortcut is (padding,)
    pad_width = ((0, 0), (kh // 2, kh // 2), (kw // 2, kh // 2))

    padded_imgs = np.pad(images, mode="constant", pad_width=pad_width)
    for i in range(h):
        for j in range(w):
            # slice region from each padded image, using kernel shape
            region = padded_imgs[:, i:(i + kh), j:(j + kw)]

            # Convolve each image (m) for this region (i, j)
            convolved[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return convolved
