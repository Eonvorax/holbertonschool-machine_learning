#!/usr/bin/env python3

"""
This is the 5-accross_the_planes module.
"""


def add_matrices2D(mat1, mat2):
    """
    Adds two matrices element-wise.
    Assuming that arr1 and arr2 are 2D matrixes of ints/floats.
    If arr1 and arr2 are not the same shape, returns None.
    """
    if len(mat1) != len(mat2):
        return None
    if any(len(row1) != len(row2) for row1, row2 in zip(mat1, mat2)):
        return None

    return [[x + y for x, y in zip(row1, row2)]
            for row1, row2 in zip(mat1, mat2)]
