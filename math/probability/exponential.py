#!/usr/bin/env python3

"""
Exponential distribution
"""


class Exponential:
    """
    Represents an exponential distribution.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes the exponential distribution.

        Parameters:
        data (list of numbers): data to be used to estimate the distribution.
        lambtha (float): expected number of occurrences in a given time frame.

        Sets the instance attribute lambtha.
        Saves lambtha as a float.

        If data is not given (i.e., None):
            - Use the given lambtha.
            - If lambtha is not a positive value, raise a ValueError with the
            message "lambtha must be a positive value".

        If data is given:
            - Calculate the lambtha of data.
            - If data is not a list, raise a TypeError with the message
            "data must be a list".
            - If data does not contain at least two data points, raise a
            ValueError with the message "data must contain multiple values".
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = 1 / (sum(data) / len(data))
