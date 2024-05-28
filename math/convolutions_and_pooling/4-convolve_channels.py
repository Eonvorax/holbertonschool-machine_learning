#!/usr/bin/env python3
"""
Convolution with Channels
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels.

    Parameters:
    - images: numpy.ndarray with shape (m, h, w, c) containing multiple
    images
    - kernel: numpy.ndarray with shape (kh, kw, c) containing the kernel
    for the convolution
    - padding: either a tuple of (ph, pw), 'same', or 'valid'
    - stride: tuple of (sh, sw)

    Returns:
    - A numpy.ndarray containing the convolved images
    """
    # Setup matrixes and padding dimensions
    m, h, w, _ = images.shape
    kh, kw, _ = kernel.shape
    sh, sw = stride

    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph = 0
        pw = 0

    # Calculate output dimensions
    output_h = (h + 2 * ph - kh) // sh + 1
    output_w = (w + 2 * pw - kw) // sw + 1

    # NOTE padding indexes: (before, after), shortcut is (padding,)
    padded_imgs = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                         mode='constant')

    # Initialize convolution output array
    convolved = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            # Extract region from padded images (added color axis)
            region = padded_imgs[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]

            # Convolve each image (m) for this region (added the color axis)
            convolved[:, i, j] = np.sum(region * kernel, axis=(1, 2, 3))

    return convolved
