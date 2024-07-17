#!/usr/bin/env python3

"""
Normal distribution
"""


class Normal:
    """
    Represents a normal distribution
    """
    # Using given approximations
    e = 2.7182818285
    pi = 3.1415926536

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

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value.

        Parameters:
        x (float): The x-value.

        Returns:
        float: The z-score of x.
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score.

        Parameters:
        z (float): The z-score.

        Returns:
        float: The x-value corresponding to z.
        """
        return self.mean + (z * self.stddev)

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value.

        Parameters:
        x (float): The x-value.

        Returns:
        float: The PDF value for x.
        """
        # Using the formula for normal distribution
        coefficient = 1 / (self.stddev * (2 * Normal.pi) ** 0.5)
        exponent = -((x - self.mean) ** 2) / (2 * self.stddev ** 2)
        return coefficient * (Normal.e ** exponent)

    def _erf(self, z):
        """
        Calculates the error function value for a given z.

        Parameters:
        z (float): The z-value.

        Returns:
        float: The error function value for z.
        """
        erf_sum = z - (z ** 3) / 3 + (z ** 5) / 10 \
            - (z ** 7) / 42 + (z ** 9) / 216
        return (2 / self.pi ** 0.5) * erf_sum

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value.

        Parameters:
        x (float): The x-value.

        Returns:
        float: The CDF value for x.
        """
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))
        return 0.5 * (1 + self._erf(z))
