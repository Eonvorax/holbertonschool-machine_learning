#!/usr/bin/env python3

"""
This is the 2-size_me_please module.
"""


def matrix_shape(matrix):
    """
    Calculates the shape of a matrix:
    Assuming all elements in the same dimension are of the same type/shape.
    The shape is returned as a list of integers.

    Iterative and recursive implementation seemed similar, so I went with
    recursion for a change.
    """
    if isinstance(matrix, list):
        return [len(matrix)] + matrix_shape(matrix[0])
    return []
