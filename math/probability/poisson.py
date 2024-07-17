#!/usr/bin/env python3

"""
Poisson distribution
"""


class Poisson:
    """
    Represents a Poisson distribution.
    """
    # Using given approximations
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes the Poisson distribution.

        Parameters:
        data (list of numbers): data to be used to estimate the distribution.
        lambtha (float): expected number of occurrences in a given time frame.

        Sets the instance attribute lambtha.
        Saves lambtha as a float.

        If data is not given (i.e., None):
            - Use the given lambtha.
            - If lambtha is not a positive value or equals to 0, raise
            a ValueError with the message "lambtha must be a positive value".

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
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of “successes”.

        Parameters:
        - `k` (int or float): The number of “successes”.

        If k is not an integer, it will be converted to an integer.
        If k is out of range (k < 0), it will return 0.

        Returns:
        - float: The PMF value for k.
        """
        e = Poisson.e
        k = int(k)
        if k < 0:
            return 0
        # PMF formula, for Poisson distribution
        return (e ** -self.lambtha) * (self.lambtha ** k) / self._factorial(k)

    def _factorial(self, n):
        """
        Calculates the factorial of a given number `n`.
        """
        if n == 0:
            return 1
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of “successes”.

        Parameters:
        k (int or float): The number of “successes”.

        If k is not an integer, it will be converted to an integer.
        If k is out of range (k < 0), it will return 0.

        Returns:
        float: The CDF value for k.
        """
        k = int(k)
        if k < 0:
            return 0

        # CDF formula for Poisson distribution: sum of PMFs, from 0 to k
        cdf_sum = 0
        for i in range(k + 1):
            cdf_sum += (Poisson.e ** -self.lambtha) * \
                (self.lambtha ** i) / self._factorial(i)

        return cdf_sum
