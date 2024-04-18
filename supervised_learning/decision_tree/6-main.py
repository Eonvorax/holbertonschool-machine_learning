#!/usr/bin/env python3

import numpy as np
Node = __import__('6-build_decision_tree').Node
Leaf = __import__('6-build_decision_tree').Leaf
Decision_Tree = __import__('6-build_decision_tree').Decision_Tree


def random_tree(max_depth, n_classes, n_features, seed=0):
    assert max_depth > 0, "max_depth must be a strictly positive integer"
    rng = np.random.default_rng(seed)
    root = Node(is_root=True, depth=0)
    root.lower = {i: -100 for i in range(n_features)}
    root.upper = {i: 100 for i in range(n_features)}

    def build_children(node):
        feat = rng.integers(0, n_features)
        node.feature = feat
        node.threshold = np.round(rng.uniform(
            0, 1)*(node.upper[feat]-node.lower[feat])+node.lower[feat], 2)
        if node.depth == max_depth-1:
            node.left_child = Leaf(
                depth=max_depth, value=rng.integers(0, n_classes))
            node.right_child = Leaf(
                depth=max_depth, value=rng.integers(0, n_classes))
        else:
            node.left_child = Node(depth=node.depth+1)
            node.left_child.lower = node.lower.copy()
            node.left_child.upper = node.upper.copy()
            node.left_child.lower[feat] = node.threshold
            node.right_child = Node(depth=node.depth+1)
            node.right_child.lower = node.lower.copy()
            node.right_child.upper = node.upper.copy()
            node.right_child.upper[feat] = node.threshold
            build_children(node.left_child)
            build_children(node.right_child)

    T = Decision_Tree(root=root)
    build_children(root)

    A = rng.uniform(
        0, 1, size=100*n_features).reshape([100, n_features])*200-100
    return T, A


T, A = random_tree(4, 3, 5, seed=1)
print(T)

T.update_predict()

print("T.pred(A) :\n", np.array([T.pred(x) for x in A]))
print("T.predict(A) :\n", T.predict(A))

test = np.all(np.equal(T.predict(A), np.array([T.pred(x) for x in A])))
print(f"Predictions are the same on the explanatory array A : {test}")
