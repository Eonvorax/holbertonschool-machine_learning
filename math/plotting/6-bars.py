#!/usr/bin/env python3

"""
This is the 6-bars module.
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    A bars graph that represents the number of fruits each person possesses.
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    # your code here
    persons = ['Farrah', 'Fred', 'Felicia']
    fruit_types = ['apples', 'bananas', 'oranges', 'peaches']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

    bottom = 0

    for i, bottom_incr in enumerate(fruit):
        plt.bar(
            persons,
            bottom_incr,
            bottom=bottom,
            label=fruit_types[i],
            color=colors[i],
            width=0.5
            )
        bottom += bottom_incr

    plt.ylabel("Quantity of Fruit")
    plt.title("Number of Fruit per Person")
    plt.yticks(np.arange(0, 81, 10))
    plt.legend()

    plt.show()
