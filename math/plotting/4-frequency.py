#!/usr/bin/env python3

"""
This is the 4-frequency module.
"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    A histogram of student scores, with bins every 10 units on the x-axis.
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # your code here
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
    plt.xlabel('Grades', fontsize=13)
    plt.ylabel('Number of Students', fontsize=13)
    plt.title('Project A', fontsize=16)
    # Had to adjust axis limits
    plt.xlim(0, 100)
    plt.ylim(0, 30)

    plt.show()
