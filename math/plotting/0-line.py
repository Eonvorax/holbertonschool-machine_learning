#!/usr/bin/env python3

"""
This is the 0-line module.
"""

import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Basic line graph.
    """

    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    # your code here
    plt.plot(y, color='red', linestyle='-')
    plt.xlim(0, 10)
    plt.show()
