#!/usr/bin/env python3

"""
This is the 13-cats_got_your_tongue module.
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenates two ndarrays along a given axis.
    Assuming that mat1 and mat2 are never empty.
    """
    return np.concatenate((mat1, mat2), axis)
