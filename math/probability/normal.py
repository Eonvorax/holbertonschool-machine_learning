#!/usr/bin/env python3

"""
Normal distribution
"""


class Normal:
    """
    Represents a normal distribution
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initializes the normal distribution.

        Parameters:
        data (list of numbers): data to be used to estimate the distribution.
        mean (float): The mean of the distribution.
        stddev (float): The standard deviation of the distribution.

        Sets the instance attributes mean and stddev.
        Saves mean and stddev as floats.

        If data is not given (i.e., None):
            - Use the given mean and stddev.
            - If stddev is not a positive value or equals to 0, raise a
            ValueError with the message "stddev must be a positive value".

        If data is given:
            - Calculate the mean and standard deviation of data.
            - If data is not a list, raise a TypeError with the message
            "data must be a list".
            - If data does not contain at least two data points, raise a
            ValueError with the message "data must contain multiple values".
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = float(sum(data) / len(data))
            # stddev is the square root of the variance
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = float(variance ** 0.5)
