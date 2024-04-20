#!/usr/bin/env python3

from sklearn import datasets
import numpy as np
Random_Forest = __import__('9-random_forest').Random_Forest


#                                     #########################
#                                     # Generating examples : #
#                                     #########################

def circle_of_clouds(n_clouds, n_objects_by_cloud, radius=1, sigma=None, seed=0, angle=0):
    """
    This function returns a dataset made of 'n_clouds' classes.
    Each class is a small gaussian cloud containing 'n_objects_by_cloud' points.
    The centers of the clouds are regularly disposed on a circle of radius 'radius' (and center (0,0)).
    The spreadth of the clouds is governed by 'sigma'.
    """
    rng = np.random.default_rng(seed)
    if not sigma:
        sigma = np.sqrt(2 - 2 * np.cos(2 * np.pi / n_clouds)) / 7

    def rotate(x, k):
        theta = 2 * k * np.pi / n_clouds + angle
        m = np.matrix([[np.cos(theta), np.sin(theta)],
                      [-np.sin(theta), np.cos(theta)]])
        return np.matmul(x, m)

    def cloud():
        return (rng.normal(size=2 * n_objects_by_cloud) * sigma).reshape(n_objects_by_cloud, 2) + np.array([radius, 0])

    def target():
        return np.array(([[i] * n_objects_by_cloud for i in range(n_clouds)]), dtype="int32").ravel()

    return np.concatenate([np.array(rotate(cloud(), k)) for k in range(n_clouds)], axis=0), target()


def iris():
    """ Returns the explanatory features and the target of the famous iris dataset """
    iris = datasets.load_iris()
    return iris.data, iris.target


def wine():
    """ Returns the explanatory features and the target of the wine dataset """
    wine = datasets.load_wine()
    return wine.data, wine.target


#                                     #########################
#                                     #    Data preparation   #
#                                     #########################

def split(explanatory, target, seed=0, proportion=.1):
    """ Returns a dictionary containing a a training dataset and a test dataset """
    rng = np.random.default_rng(seed)
    test_indices = rng.choice(target.size, int(
        target.size * proportion), replace=False)
    test_filter = np.zeros_like(target, dtype="bool")
    test_filter[test_indices] = True

    return {"train_explanatory": explanatory[np.logical_not(test_filter), :],
            "train_target": target[np.logical_not(test_filter)],
            "test_explanatory": explanatory[test_filter, :],
            "test_target": target[test_filter]}


# Main 1
for d, name in zip([split(*circle_of_clouds(10, 30)), split(*iris()), split(*wine())], ["circle of clouds", "iris dataset", "wine dataset"]):
    print("-"*52+"\n"+name+" :")
    F = Random_Forest(max_depth=6)
    F.fit(d["train_explanatory"], d["train_target"], verbose=1)
    print(
        f"    - Accuracy on test          : {F.accuracy(d['test_explanatory'],d['test_target'])}")
print("-"*52)
