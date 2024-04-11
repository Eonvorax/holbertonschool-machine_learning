#!/usr/bin/env python3

"""
This is the 8-ridin_bareback module.
"""


def mat_mul(mat1, mat2):
    """
    Performs matrix multiplication.
    Assuming all elements in the same dimension are of the same type/shape
    and all elements are ints/floats.
    Return None if the matrixes can't be multiplied.
    """
    if len(mat1[0]) != (len(mat2)):
        return None

    # Not the most efficient solution, could use zip and comprehensions
    # NOTE lookup "matrix transposition cache locality" for efficiency
    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            elt = 0
            for k in range(len(mat2)):
                elt += mat1[i][k] * mat2[k][j]
            row.append(elt)
        result.append(row)

    return result
