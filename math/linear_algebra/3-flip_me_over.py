#!/usr/bin/env python3

"""
This is the 3-flip_me_over module.
"""


def matrix_transpose(matrix):
    """
    Returns the transpose of a 2D matrix, matrix:
    Assuming that matrix is never empty.
    Assuming all elements in the same dimension are of the same type/shape.
    """
    # Pre-counting rows and columns
    n_rows = len(matrix)
    n_cols = len(matrix[0])

    # Using a comprehension to avoid having to pre-initialize transpose
    return [[matrix[j][i] for j in range(n_rows)] for i in range(n_cols)]
