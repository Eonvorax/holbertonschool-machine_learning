#!/usr/bin/env python3

"""
This the 101-the_whole_barn module.
"""


def add_matrices(mat1, mat2):
    """
    Add two matrices element-wise.

    Args:
    - mat1: list, the first matrix to add.
    - mat2: list, the second matrix to add.

    Returns:
    - list: a new matrix containing the element-wise sum of mat1 and mat2,
    or None if matrices have different shapes.
    """
    # Checking if matrices have the same shape (tuple)
    if get_matrix_shape(mat1) != get_matrix_shape(mat2):
        return None

    result = []

    # Performing element-wise addition in current dimension
    for item1, item2 in zip(mat1, mat2):
        # Check if item is a list (a nested matrix)
        if isinstance(item1, list):
            # Recursively call add_matrices on found nested matrix
            result.append(add_matrices(item1, item2))
        else:
            # Add simple number elements
            result.append(item1 + item2)

    return result


def get_matrix_shape(matrix):
    """
    Get the shape of a matrix by recursively checking the length and type of
    its elements.

    Args:
    - list: the matrix

    Returns:
    - tuple: the shape of the matrix.
    """
    # Check if item is a list (a nested matrix)
    if isinstance(matrix[0], list):
        # Recursively call get_matrix_shape on found nested matrix
        return (len(matrix),) + get_matrix_shape(matrix[0])
    else:
        # No nested matrix; returning length current matrix.
        return (len(matrix),)
