#!/usr/bin/env python3

"""
Binomial distribution
"""


class Binomial:
    """
    Represents a binomial distribution
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializes the binomial distribution.

        Parameters:
        data (list of numbers): data to be used to estimate the distribution.
        n (int): The number of Bernoulli trials.
        p (float): The probability of a “success”.

        Sets the instance attributes n and p.
        Saves n as an integer and p as a float.

        If data is not given (i.e., `None`):
            - Use the given n and p.
            - If n is not a positive value, raise a ValueError with the
            message "n must be a positive value".
            - If p is not a valid probability, raise a ValueError with the
            message "p must be greater than 0 and less than 1".

        If data is given:
            - Calculate n and p from data.
            - Round n to the nearest integer (rounded, not casting).
            - Calculate p first and then calculate n. Then recalculate p.
            - If data is not a list, raise a TypeError with the message
            "data must be a list".
            - If data does not contain at least two data points, raise a
            ValueError with the message "data must contain multiple values".
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            p_est = 1 - (variance / mean)
            n_est = round(mean / p_est)
            p_est = mean / n_est

            # Using recalculated p value
            self.n = n_est
            self.p = p_est

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

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of “successes”.

        Parameters:
        k (int): The number of “successes”.

        Returns:
        float: The PMF value for k.
        If k is out of range (k < 0 or k > n), returns 0.
        """
        if not isinstance(k, int):
            k = int(k)

        if k < 0 or k > self.n:
            return 0

        # binomial coefficient
        nck = self._factorial(self.n) / (self._factorial(k)
                                         * self._factorial(self.n - k))
        # Use binomial coeff. to calculate the PMF
        return nck * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of “successes”.

        Parameters:
        k (int): The number of “successes”.

        Returns:
        float: The CDF value for k.
        If k is out of range (k < 0 or k > n), returns 0.
        """
        if not isinstance(k, int):
            k = int(k)

        if k < 0 or k > self.n:
            return 0

        # As before, the CDF is the sum of PMFs for each "success"
        return sum(self.pmf(i) for i in range(k + 1))
