#!/usr/bin/env python3
"""
Pooling: avg or max
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.

    Parameters:
    - images: numpy.ndarray with shape (m, h, w, c) containing images
    - kernel_shape: tuple of (kh, kw) containing the kernel shape
    - stride: tuple of (sh, sw)
    - mode: indicates the type of pooling ('max' or 'avg')

    Returns:
    - A numpy.ndarray containing the pooled images
    """
    # Setup stride and matrix dimensions
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate output dimensions
    output_h = (h - kh) // sh + 1
    output_w = (w - kw) // sw + 1

    # Choose pooling function based on given mode
    if mode == 'max':
        pooling_func = np.max
    elif mode == 'avg':
        pooling_func = np.mean

    # Initialize pooling output array
    pooled = np.zeros((m, output_h, output_w, c))

    for i in range(output_h):
        for j in range(output_w):
            # Extract region from images
            region = images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            # Pooling on each channel of this region
            pooled[:, i, j, :] = pooling_func(region, axis=(1, 2))

    return pooled
