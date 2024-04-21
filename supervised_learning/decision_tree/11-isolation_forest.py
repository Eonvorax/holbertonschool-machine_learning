#!/usr/bin/env python3

"""
This is the 11-isolation_forest module.
"""

import numpy as np
Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree


class Isolation_Random_Forest():
    """
    Random forest class, using Isolation Trees.
    """
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def predict(self, explanatory):
        """
        Returns an array of the average prediction for each tree in
        numpy_preds, based on the given explanatory variables.
        """
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return predictions.mean(axis=0)

    def fit(self, explanatory, n_trees=100, verbose=0):
        """
        Fit the Isolation Forest model to the given explanatory variables.

        Args:
            explanatory (numpy.ndarray): The explanatory variables.
            n_trees (int): The number of trees in the forest (default=100).
            verbose (int): Verbosity mode. 0 = silent, 1 = verbose (default=0).

        Returns:
            None
        """
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        for i in range(n_trees):
            T = Isolation_Random_Tree(
                max_depth=self.max_depth, seed=self.seed+i)
            T.fit(explanatory)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : { np.array(depths).mean()      }
    - Mean number of nodes           : { np.array(nodes).mean()       }
    - Mean number of leaves          : { np.array(leaves).mean()      }""")

    def suspects(self, explanatory, n_suspects):
        """
        Returns the n_suspects rows in explanatory that have the
        smallest mean depth.
        """
        depths = self.predict(explanatory)
        # NOTE Getting indices that would sort the depths array in asc. order
        # (sorted based on their predicted depth)
        sorted_indices = np.argsort(depths)

        # Using these indices to get the corresponding suspect rows in
        # explanatory (the dataset) and their depths (the predictions)
        return explanatory[sorted_indices[:n_suspects]], \
            depths[sorted_indices[:n_suspects]]
