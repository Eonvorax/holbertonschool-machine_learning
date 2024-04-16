#!/usr/bin/env python3

"""
This is the 1-build_decision_tree module.
"""

import numpy as np


class Node:
    """
    Represents a node in a decision tree.

    Attributes:
        feature (int): The index of the feature used to split at this node.
        threshold (float): The threshold value used to split at this node.
        left_child (Node): The left child node.
        right_child (Node): The right child node.
        is_leaf (bool): Indicates whether this node is a leaf node.
        is_root (bool): Indicates whether this node is the root node.
        sub_population (None or ndarray): The subset of the population that
        reaches this node.
        depth (int): The depth of this node in the decision tree.
    """

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Recursively calculates the maximum depth of the subtree below
        this node.

        Returns:
            int: The maximum depth below this node.
        """
        if self.is_leaf:
            return self.depth

        return max(self.left_child.max_depth_below(),
                   self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """
        Returns the number of nodes under this node.
        If only_leaves is True, only counts leaf nodes.
        """
        if only_leaves and self.is_leaf:
            return 1

        if not self.is_leaf:
            # NOTE Counting the current node only if only_leaves == False
            return self.left_child.count_nodes_below(only_leaves=only_leaves)\
                + self.right_child.count_nodes_below(only_leaves=only_leaves)\
                + (not only_leaves)


class Leaf(Node):
    """
    Represents a leaf node in a decision tree.

    Attributes:
        value (any): The value associated with the leaf node.
        is_leaf (bool): Indicates whether the node is a leaf node.
        depth (int): The depth of the leaf node in the decision tree.
    """

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Returns the maximum depth below the leaf node.

        Returns:
            int: The maximum depth below the leaf node.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Overwrites the same method for the Node class.
        Returns 1.
        """
        return 1


class Decision_Tree():
    """
    A class representing a decision tree.

    Attributes:
        max_depth (int): The maximum depth of the decision tree.
        min_pop (int): The minimum population required to split a node.
        seed (int): The seed value for random number generation.
        split_criterion (str): The criterion used for splitting nodes.
        root (Node): The root node of the decision tree.
        explanatory: The explanatory variable(s) used for prediction.
        target: The target variable used for prediction.
        predict: The prediction function used for making predictions.

    Methods:
        depth(): Returns the maximum depth of the decision tree.
    """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        Returns the maximum depth of the decision tree.

        Returns:
            int: The maximum depth of the decision tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Returns the number of nodes in the decision tree.
        If only_leaves is True, only counts leaf nodes.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)
