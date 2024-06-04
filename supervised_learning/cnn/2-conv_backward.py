#!/usr/bin/env python3
"""
Convolutional Back Prop
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network.

    Parameters:
    dZ (numpy.ndarray): Partial derivatives with respect to the unactivated
        output of the convolutional layer, with shape (m, h_new, w_new, c_new)
    A_prev (numpy.ndarray): Output of the previous layer,
        with shape (m, h_prev, w_prev, c_prev)
    W (numpy.ndarray): Kernels for the convolution,
        with shape (kh, kw, c_prev, c_new)
    b (numpy.ndarray): Biases applied to the convolution,
        with shape (1, 1, 1, c_new)
    padding (str): Padding used, either 'same' or 'valid' (default is 'same')
    stride (tuple): Strides for the convolution, with shape (sh, sw)

    Returns:
    tuple: (dA_prev, dW, db)
        - dA_prev (numpy.ndarray): Partial derivatives with respect to
            the previous layer
        - dW (numpy.ndarray): Partial derivatives with respect to
            the kernels
        - db (numpy.ndarray): Partial derivatives with respect to
            the biases
    """
    # Setup stride, matrixes and padding dimensions
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2 + 1
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2 + 1
    elif padding == 'valid':
        ph = 0
        pw = 0

    # Padding input as needed
    padded_A_prev = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant')

    # Initialize derivative arrays
    dA = np.zeros(shape=padded_A_prev.shape)
    dW = np.zeros(shape=W.shape)
    # NOTE db can just be calculated directly
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(m):  # Examples (images)
        for h in range(h_new):  # heights
            for w in range(w_new):  # widths
                for c in range(c_new):  # channels
                    # Prepare slice indexes to account for stride
                    v_start = h * sh
                    v_end = v_start + kh
                    h_start = w * sw
                    h_end = h_start + kw
                    # Update gradients for this channel
                    dA[i, v_start:v_end, h_start:h_end, :] +=\
                        W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] +=\
                        padded_A_prev[i, v_start:v_end, h_start:h_end, :]\
                        * dZ[i, h, w, c]

    if padding == 'same':
        # Slice off the extra padding if same padding was used
        dA = dA[:, ph:-ph, pw:-pw, :]

    return dA, dW, db
