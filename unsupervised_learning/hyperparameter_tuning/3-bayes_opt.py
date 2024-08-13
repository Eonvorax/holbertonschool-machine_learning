#!/usr/bin/env python3
"""
Bayesian Optimization
"""

import numpy as np
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
