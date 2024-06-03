#!/usr/bin/env python3
"""
Convolutional Forward Prop
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a CNN.

    Parameters:
    A_prev (numpy.ndarray): Output of previous layer, of shape
    (m, h_prev, w_prev, c_prev)
        - m (int): Number of examples
        - h_prev (int): Height of the previous layer
        - w_prev (int): Width of the previous layer
        - c_prev (int): Number of channels in the previous layer
    W (numpy.ndarray): Kernels, with shape (kh, kw, c_prev, c_new)
        - kh (int): Filter height
        - kw (int): Filter width
        - c_prev (int): Number of channels in the previous layer
        - c_new (int): Number of channels in the output
    b (numpy.ndarray): Biases, of shape (1, 1, 1, c_new)
    activation (function): Activation function
    padding (str): Padding used, either 'same' or 'valid' (default is 'same')
    stride (tuple): Strides for the convolution, with shape (sh, sw)
        - sh (int): Stride for the height
        - sw (int): Stride for the width

    Returns:
    numpy.ndarray: The output of the convolutional layer
    """
    # Setup stride, matrixes and padding dimensions
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2 + 1
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2 + 1
    elif padding == 'valid':
        ph = 0
        pw = 0

    # Calculate output dimensions
    output_h = (h_prev + 2 * ph - kh) // sh + 1
    output_w = (w_prev + 2 * pw - kw) // sw + 1

    # Padding input as needed
    padded_imgs = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                         mode='constant')

    # Initialize convolution output array
    convolved = np.zeros((m, output_h, output_w, c_new))

    for i in range(output_h):
        for j in range(output_w):
            # Extract region from padded images
            region = padded_imgs[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            for k in range(c_new):
                # Convolve each image (m) in the region, using kernel k
                convolved[:, i, j, k] = np.sum((region * W[:, :, :, k]),
                                               axis=(1, 2, 3))

    # Layer l activation output: A(l+1) = g(Z), with Z = A(l) * W + b
    return activation(convolved + b)
