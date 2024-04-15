#!/usr/bin/env python3

"""
This is the 10-matisse module.
"""


def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial coefficient list.
    """
    if not isinstance(poly, list) \
            or len(poly) == 0 \
            or not all(isinstance(coeff, (int, float)) for coeff in poly):
        return None

    if len(poly) == 1:
        return [0]

    return [coef * i for i, coef in enumerate(poly) if i != 0]
