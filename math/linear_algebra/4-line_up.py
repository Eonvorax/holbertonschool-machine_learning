#!/usr/bin/env python3

"""
This is the 4-line_up module.
"""


def add_arrays(arr1, arr2):
    """
    Adds two arrays element-wise and returns the new list.
    Assuming that arr1 and arr2 are lists of ints/floats.
    If arr1 and arr2 are not the same shape, returns None.
    """
    if len(arr1) != len(arr2):
        return None

    # NOTE zip() refresher : iterates an index and returns tuples of elements
    # from the given iterators (at the index) until at least one given
    # iterator is exhausted.
    return [x_arr1 + y_arr2 for x_arr1, y_arr2 in zip(arr1, arr2)]
