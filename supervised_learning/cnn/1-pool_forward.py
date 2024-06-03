#!/usr/bin/env python3
"""
Pooling Forward Prop
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a CNN.

    Parameters:
    - A_prev (numpy.ndarray): Output of previous layer, of shape
    (m, h_prev, w_prev, c_prev)
        - m (int): Number of examples
        - h_prev (int): Height of the previous layer
        - w_prev (int): Width of the previous layer
        - c_prev (int): Number of channels in the previous layer
    - kernel_shape: tuple of (kh, kw) containing the kernel shape
    - stride: tuple of (sh, sw), the stride height and width
    - mode: indicates the type of pooling ('max' or 'avg')

    Returns:
    numpy.ndarray: The output of the pooling layer
    """
    # Setup stride and matrix dimensions
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate output dimensions
    output_h = (h_prev - kh) // sh + 1
    output_w = (w_prev - kw) // sw + 1

    # Choose pooling function based on given mode
    if mode == 'max':
        pooling_func = np.max
    elif mode == 'avg':
        pooling_func = np.mean

    # Initialize pooling output array
    pooled = np.zeros((m, output_h, output_w, c_prev))

    for i in range(output_h):
        for j in range(output_w):
            # Extract region from input
            region = A_prev[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            # Pooling on each channel of this region
            pooled[:, i, j, :] = pooling_func(region, axis=(1, 2))

    return pooled
