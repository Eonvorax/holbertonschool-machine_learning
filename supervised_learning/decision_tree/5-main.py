#!/usr/bin/env python3

import numpy as np
Node = __import__('5-build_decision_tree').Node
Leaf = __import__('5-build_decision_tree').Leaf
Decision_Tree = __import__('5-build_decision_tree').Decision_Tree


def example_0():
    leaf0 = Leaf(0, depth=1)
    leaf1 = Leaf(0, depth=2)
    leaf2 = Leaf(1, depth=2)
    internal_node = Node(feature=1, threshold=30000,
                         left_child=leaf1, right_child=leaf2, depth=1)
    root = Node(feature=0, threshold=.5, left_child=leaf0,
                right_child=internal_node, depth=0, is_root=True)
    return Decision_Tree(root=root)


def example_1(depth):
    level = [Leaf(i, depth=depth) for i in range(2 ** depth)]
    level.reverse()

    def get_v(node):
        if node.is_leaf:
            return node.value
        else:
            return node.threshold

    for d in range(depth):
        level = [Node(feature=0,
                      threshold=(get_v(level[2 * i]) +
                                 get_v(level[2 * i + 1])) / 2,
                      left_child=level[2 * i],
                      right_child=level[2 * i + 1], depth=depth - d - 1) for i in range(2 ** (depth - d - 1))]
    root = level[0]
    root.is_root = True
    return Decision_Tree(root=root)


def print_indicator_values_on_leaves(T, A):
    leaves = T.get_leaves()
    T.update_bounds()
    for leaf in leaves:
        leaf.update_indicator()
    print("values of indicators of leaves :\n", np.array(
        [leaf.indicator(A) for leaf in leaves]))


T = example_0()
A = np.array([[1, 22000], [1, 44000], [0, 22000], [0, 44000]])
print("\n\nFor example_0()")
print("A=\n", A)
print_indicator_values_on_leaves(T, A)

T = example_1(4)
A = np.array([[11.65], [6.917]])
print("\n\nFor example_1(4)")
print("A=\n", A)
print_indicator_values_on_leaves(T, A)
