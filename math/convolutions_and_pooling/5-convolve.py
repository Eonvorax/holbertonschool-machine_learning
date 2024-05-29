#!/usr/bin/env python3
"""
Multiple Kernels Convolution
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels.

    Parameters:
    - images: numpy.ndarray of shape (m, h, w, c) containing multiple images
    - kernels: numpy.ndarray of shape (kh, kw, c, nc) containing the kernels
    - padding: either a tuple of (ph, pw), 'same', or 'valid'
    - stride: tuple of (sh, sw)

    Returns:
    - A numpy.ndarray containing the convolved images
    """
    # Setup stride, matrixes and padding dimensions
    m, h, w, _ = images.shape
    kh, kw, _, nc = kernels.shape
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
    convolved = np.zeros((m, output_h, output_w, nc))

    for i in range(output_h):
        for j in range(output_w):
            # Extract region from padded images (with color axis)
            region = padded_imgs[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            for k in range(nc):
                # Convolve each image (m) for this region, using the
                # corresponding kernel (k) for this channel
                convolved[:, i, j, k] = np.sum(region * kernels[:, :, :, k],
                                               axis=(1, 2, 3))

    return convolved
