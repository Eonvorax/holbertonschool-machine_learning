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

    def __str__(self):
        """
        Prints string representation of the node and its children.
        """

        if self.is_root:
            s = "root"
        else:
            s = "-> node"

        return f"{s} [feature={self.feature}, threshold={self.threshold}]\n"\
            + self.left_child_add_prefix(str(self.left_child))\
            + self.right_child_add_prefix(str(self.right_child))

    def left_child_add_prefix(self, text):
        """
        Adds the string representation of the left child to the given text
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """
        Adds the string representation of the right child to the given text
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        # NOTE Had to strip the extra newline after right node
        # There may be better alternatives
        return (new_text.rstrip())

    def get_leaves_below(self):
        """
        Returns the list of all leaves below this one.
        """

        return self.left_child.get_leaves_below()\
            + self.right_child.get_leaves_below()

    def update_bounds_below(self):
        """
        Recursively compute, for each node, two dictionaries stored as
        attributes Node.lower and Node.upper.
        """
        if self.is_root:
            self.lower = {0: -1 * np.inf}
            self.upper = {0: np.inf}

        for child in [self.left_child, self.right_child]:
            child.upper = self.upper.copy()
            child.lower = self.lower.copy()

        if self.feature in self.left_child.lower.keys():
            # Updating with lower left threshold
            self.left_child.lower[self.feature] = \
                max(self.threshold, self.left_child.lower[self.feature])
        else:
            self.left_child.lower[self.feature] = self.threshold

        if self.feature in self.right_child.upper.keys():
            # Updating with upper right threshold
            self.right_child.upper[self.feature] = \
                min(self.threshold, self.right_child.upper[self.feature])
        else:
            self.right_child.upper[self.feature] = self.threshold

        self.left_child.update_bounds_below()
        self.right_child.update_bounds_below()

    def update_indicator(self):
        """
        Update the indicator function based on the lower and upper bounds.

        The indicator function is a lambda function that takes in a 2D numpy
        array `x` representing the features of the individuals and returns a
        1D numpy array of size `n_individuals` where the `i`-th element is
        `True` if the `i`-th individual satisfies the conditions specified
        by the lower and upper bounds.
        """
        def is_large_enough(x):
            return np.all(
                np.array([np.greater(x[:, key], self.lower[key])
                          for key in self.lower]), axis=0
            )

        def is_small_enough(x):
            return np.all(
                np.array([np.less_equal(x[:, key], self.upper[key])
                          for key in self.upper]), axis=0
            )

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        """
        Predicts the class label for a given input sample.

        Args:
            x (list): The input sample to predict the class label for.

        Returns:
            int: The predicted class label for the input sample (leaf value)
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


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

    def __str__(self):
        # NOTE had to add that typo empty space at the end
        # checker repo with the mistake : malekmrabti213
        return (f"-> leaf [value={self.value}] ")

    def get_leaves_below(self):
        """
        Returns this leaf as a list element.
        """
        return [self]

    def update_bounds_below(self):
        """
        Does nothing ?
        """
        pass

    def pred(self, x):
        """
        Returns the leaf's value.
        """
        return self.value


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

    def __str__(self):
        # NOTE cleaner to update this than use the "solution" I've seen
        return f"{self.root.__str__()}\n"

    def get_leaves(self):
        """
        Gets the list of leaves in the tree.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Calls update_bounds_below().
        """
        self.root.update_bounds_below()

    def pred(self, x):
        """
        Predicts the class label for a given input sample.
        Starts the recursion from the root.

        Args:
            x (array-like): The input sample to be classified.

        Returns:
            The predicted class label for the input sample.
        """
        return self.root.pred(x)

    def update_predict(self):
        """
        Update the prediction function of the decision tree.

        This method updates the prediction function of the decision tree
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
