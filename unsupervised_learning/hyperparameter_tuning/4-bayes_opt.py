#!/usr/bin/env python3
"""
Bayesian Optimization - Acquisition
"""

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    This class represents the Bayesian Optimization technique
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """
        Initializes the Bayesian Optimization.

        Parameters:
        - f: The black-box function to be optimized.
        - X_init: numpy.ndarray of shape (t, 1), inputs already sampled.
        - Y_init: numpy.ndarray of shape (t, 1), outputs of the black-box
        function for each input.
        - bounds: tuple of (min, max), bounds of the space for the optimal
        point.
        - ac_samples: int, the number of samples for acquisition.
        - l: float, the length parameter for the kernel.
        - sigma_f: float, the standard deviation of the output.
        - xsi: float, the exploration-exploitation factor.
        - minimize: bool, determines whether to perform minimization or
        maximization.
        """
        self.f = f
        self.gp = GP(X_init=X_init, Y_init=Y_init, l=l, sigma_f=sigma_f)
        # generate an array of evenly-spaced sampling values in the bounds
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculate the next best sample location using the Expected Improvement
        (EI) acquisition function.

        The acquisition function balances exploration and exploitation by
        considering the expected improvement over the best current observation,
        either for minimization or maximization depending on the problem setup.

        Returns:
        --------
        - X_next : numpy.ndarray of shape (1,), next best sample point to
        evaluate the black-box function.

        - EI : numpy.ndarray of shape (ac_samples,) the expected improvement
        values for each point in the acquisition sample points (X_s).
        """
        # Predict mean and variance for each point in X_s
        mu_s, sigma_s = self.gp.predict(self.X_s)

        # Minimization or maximization, depending on predefined boolean
        if self.minimize:
            Y_best = np.min(self.gp.Y)
            improvement = Y_best - mu_s - self.xsi
        else:
            Y_best = np.max(self.gp.Y)
            improvement = mu_s - Y_best - self.xsi

        # Calc. Z and Expected Improvement
        Z = improvement / sigma_s
        EI = improvement * norm.cdf(Z) + sigma_s * norm.pdf(Z)

        # Finding the next best point to sample
        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI
