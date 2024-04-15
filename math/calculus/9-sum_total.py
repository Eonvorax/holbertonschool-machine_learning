#!/usr/bin/env python3

"""
This is the sum_total module.
"""


def summation_i_squared(n):
    """
    Calculates the sum of squared integers from 1 to n.
    """
    if not isinstance(n, int) or n < 1:
        return None

    # NOTE formula for the sum of the squares of the first n natural numbers
    return n * (n + 1) * (2 * n + 1) // 6
