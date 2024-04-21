#!/usr/bin/env python3

"""
This is the 10-isolation_tree module.
"""

import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree():
    """
    A class representing an Isolation tree.
    """
    def __init__(self, max_depth=10, seed=0, root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """
        String representation for the tree object.
        """
        # NOTE Might need to remove the newline for Isolation_Tree
        return f"{self.root.__str__()}\n"

    def depth(self):
        """
        Returns the maximum depth of the isolation tree.

        Returns:
            int: The maximum depth of the isolation tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Returns the number of nodes in the isolation tree.
        If only_leaves is True, only counts leaf nodes.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """
        Calls update_bounds_below().
        """
        self.root.update_bounds_below()

    def get_leaves(self):
        """
        Gets the list of leaves in the tree.
        """
        return self.root.get_leaves_below()

    def update_predict(self):
        """
        Update the prediction function of the isolation tree.

        This method updates the prediction function of the isolation tree
        by updating the indicators of all the leaves and creating a new
        prediction function based on the updated indicators.
        Results in an array of predictions for
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        # Setting up self.predict like a function, with array A as input
        self.predict = lambda A: np.array([self.root.pred(x) for x in A])

    def np_extrema(self, arr):
        """
        Returns extrema of array.
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        Randomly selects a feature and threshold to split the node's
        subpopulation.

        Args:
            node (Node): The node to split.

        Returns:
            tuple: A tuple containing the selected feature and threshold.
        """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """
        Returns a leaf child node with the given depth and subpopulation.
        """
        leaf_child = Leaf(node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        Create a new child node for the given parent node.

        Args:
            node (Node): The parent node.
            sub_population (list): The sub-population associated with
                the child node.

        Returns:
            Node: The newly created child node.
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """
        Fits a tree node by recursively splitting the data based on
        the best split criterion.
        """
        node.feature, node.threshold = self.random_split_criterion(node)

        max_criterion = np.greater(
            self.explanatory[:, node.feature],
            node.threshold)

        left_population = np.logical_and(
            node.sub_population,
            max_criterion)
        # "War does not determine who is right - only who is left."
        right_population = np.logical_and(
            node.sub_population,
            np.logical_not(max_criterion))

        # Is left node a leaf ? If depth is max_depth, yes
        is_left_leaf = np.any(np.array(
            [node.depth == self.max_depth - 1,
             np.sum(left_population) <= self.min_pop]))

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf ?
        is_right_leaf = np.any(np.array(
            [node.depth == self.max_depth - 1,
             np.sum(right_population) <= self.min_pop]))

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """
        Initializes some attributes of the tree and then calls a new method
        Isolation_Random_Tree.fit_node on the root
        """
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones_like(
            explanatory.shape[0], dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : { self.depth()       }
    - Number of nodes           : { self.count_nodes() }
    - Number of leaves          : { self.count_nodes(only_leaves=True) }""")
