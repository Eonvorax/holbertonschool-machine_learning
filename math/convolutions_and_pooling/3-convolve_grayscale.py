#!/usr/bin/env python3
"""
Strided Convolution
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images.

    Parameters:
    - images: numpy.ndarray with shape (m, h, w) containing multiple
    grayscale images
    - kernel: numpy.ndarray with shape (kh, kw) containing the kernel
    for the convolution
    - padding: either a tuple of (ph, pw), 'same', or 'valid'
    - stride: tuple of (sh, sw)

    Returns:
    - A numpy.ndarray containing the convolved images
    """
    # Setup matrixes and padding dimensions
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == "same":
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2
    elif padding == "valid":
        ph = 0
        pw = 0
    else:
        ph, pw = padding

    # Calculate output dimensions
    output_h = (h + 2 * ph - kh) // sh + 1
    output_w = (w + 2 * pw - kw) // sw + 1

    # NOTE padding indexes: (before, after), shortcut is (padding,)
    padded_imgs = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    # Initialize convolution output array
    convolved = np.zeros((m, output_h, output_w))

    for i in range(0, output_h):
        for j in range(0, output_w):
            # Extract region from padded images, scaling indexes by stride
            region = padded_imgs[:, i*sh:i*sh+kh, j*sw:j*sw+kw]

            # Convolve each image (m) for this region (i, j)
            convolved[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return convolved
