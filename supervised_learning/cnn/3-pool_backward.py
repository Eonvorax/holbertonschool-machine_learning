#!/usr/bin/env python3
"""
Pooling Back Prop
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network.

    Parameters:
        dA (numpy.ndarray): Partial derivatives with respect to the output
        of the pooling layer, with shape (m, h_new, w_new, c_new)
        A_prev (numpy.ndarray): Output of the previous layer, with
        shape (m, h_prev, w_prev, c)
        kernel_shape (tuple): Size of the kernel for the pooling, with
        shape (kh, kw)
        stride (tuple): Strides for the pooling, with shape (sh, sw)
        mode (str): Type of pooling, either 'max' or 'avg'

    Returns:
        numpy.ndarray: Partial derivatives with respect to the previous
        layer (dA_prev)
    """
    # Setup stride, matrixes and padding dimensions
    m, h_new, w_new, c = dA.shape
    sh, sw = stride
    kh, kw = kernel_shape

    # Initialize derivatives array
    dA_prev = np.zeros(shape=A_prev.shape)

    for i in range(m):  # Examples (images)
        for h in range(h_new):  # heights
            for w in range(w_new):  # widths
                for f in range(c):  # channels
                    # Prepare slice indexes to account for stride
                    v_start = h * sh
                    v_end = v_start + kh
                    h_start = w * sw
                    h_end = h_start + kw
                    # Update gradients for this channel
                    if mode == 'avg':
                        avg_dA = dA[i, h, w, f] / kh / kw
                        dA_prev[i, v_start:v_end, h_start:h_end, f] +=\
                            (np.ones((kh, kw)) * avg_dA)
                    elif mode == 'max':
                        region = A_prev[i, v_start:v_end, h_start:h_end, f]
                        mask = (region == np.max(region))
                        dA_prev[i, v_start:v_end, h_start:h_end, f] +=\
                            mask * dA[i, h, w, f]

    return dA_prev
