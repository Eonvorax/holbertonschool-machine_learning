#!/usr/bin/env python3

"""
This is the 12-bracin_the_elements module.
"""


def np_elementwise(mat1, mat2):
    """
    Performs element-wise operations.
    Returns a tuple containing the element-wise result of the operation.
    """
    # NOTE operations on ndarrays are done element-wise by default
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
