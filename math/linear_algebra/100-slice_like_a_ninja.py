#!/usr/bin/env python3

"""
This is the 100-slice_like_a_ninja module.
"""


def np_slice(matrix, axes={}):
    """
    Slice a matrix along specific axes.

    Args:
    - matrix: numpy.ndarray, the input matrix to be sliced.
    - axes: dict, where the key is an axis to slice along and the value
    is a tuple representing the slice to make along that axis.

    Returns:
    - numpy.ndarray: a new matrix sliced according to the provided axes.
    """
    # Creating a copy of given matrix to preserve the original
    sliced_matrix = matrix.copy()

    # Iterate over the axes dictionary, keep slicing on the corresponding axis
    for axis, slice_tuple in axes.items():
        # NOTE First drafts were a mess, detailed it with comments for clarity:
        # Base slice tuple : all elements along the current axis
        base_slice = (slice(None),) * axis
        # New slice tuple using the specific slice from the axes dictionary
        specific_slice = (slice(*slice_tuple),)
        # Combine base slice tuple with the specific slice tuple
        axis_slice = base_slice + specific_slice
        # And finally, slice using the tuple
        sliced_matrix = sliced_matrix[axis_slice]

    return sliced_matrix
