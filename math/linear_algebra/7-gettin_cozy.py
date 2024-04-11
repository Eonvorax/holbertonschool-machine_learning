#!/usr/bin/env python3

"""
This is the 6-gettin_cozy module.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis:

    Assuming that mat1 and mat2 are 2D matrices containing ints/floats
    Assuming all elements in the same dimension are the same type/shape
    You must return a new matrix
    If the two matrices cannot be concatenated, return None.
    """

    if axis == 0:
        # Checking length of rows (assuming they're all the same length)
        if len(mat1[0]) != len(mat2[0]):
            return None
        # Axis at default value of 0, inserting mat2 at the end of mat1
        # NOTE Making shallow copies of rows from each matrix to preserve them
        return [row[:] for row in mat1] + [row[:] for row in mat2]
    elif axis == 1:
        # Checking matrix length
        if len(mat1) != len(mat2):
            return None
        # Inserting mat1 elements vertically (at the end of mat1's rows)
        return [mat1[i] + mat2[i] for i in range(len(mat1))]
    else:
        # Since this only works for axis 0 an 1, other values default to:
        return None
