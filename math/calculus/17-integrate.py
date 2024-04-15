#!/usr/bin/env python3

"""
This is the 17-integrate module.
"""


def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial coefficient list.
    """
    if not isinstance(poly, list) \
            or len(poly) == 0 \
            or not all(isinstance(coeff, (int, float)) for coeff in poly):
        return None

    if not isinstance(C, (int, float)):
        return None

    # Special case: just a constant
    if len(poly) == 1:
        return [C]

    integral = [coeff / (i + 1) for i, coeff in enumerate(poly)]
    integral = [int(coeff) if coeff.is_integer() else coeff
                for coeff in integral]
    integral.insert(0, C)

    return integral
