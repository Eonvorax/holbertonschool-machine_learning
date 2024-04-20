#!/usr/bin/env python3

"""
This is the 9-random_forest module, relying on module
8-build_decision_tree.
"""

Decision_Tree = __import__('8-build_decision_tree').Decision_Tree
import numpy as np


class Random_Forest():
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """
        Returns of array of the most frequent prediction for
        each tree in numpy_preds, based on given explanatory.
        """
        # Generate predictions for each tree in the forest
        predictions = np.array(
            [tree_predict(explanatory) for tree_predict in self.numpy_preds])

        # Calculate the mode (most frequent) prediction for each example
        mode_predictions = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), # Mode
            axis=0, # Axis is 0 since it's a 1D array
            arr=predictions) # Applied to array of predictions

        return mode_predictions

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []
        for i in range(n_trees):
            T = Decision_Tree(max_depth=self.max_depth,
                              min_pop=self.min_pop, seed=self.seed+i)
            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : { np.array(depths).mean()      }
    - Mean number of nodes           : { np.array(nodes).mean()       }
    - Mean number of leaves          : { np.array(leaves).mean()      }
    - Mean accuracy on training data : { np.array(accuracies).mean()  }
    - Accuracy of the forest on td   : {self.accuracy(self.explanatory,self.target)}""")

    def accuracy(self, test_explanatory, test_target):
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target)) / test_target.size
