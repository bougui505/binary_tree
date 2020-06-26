#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-06-26 09:08:41 (UTC+0200)

import numpy
import collections


def extend_arr_tree(arr, index, val):
    """
    Add given val at given ind in arr
    """
    assert index > len(arr) - 1
    arr.extend([None, ] * (index - len(arr)))
    arr.append(val)


class Tree(object):
    def __init__(self):
        self.arr = [0, ]  # Array data structure for the tree

    def add_child(self, parent, child):
        assert parent in set(self.arr), f"{parent} is not present in the tree"
        assert child not in set(self.arr), f"{child} already in tree"
        i = self.arr.index(parent)
        if len(self.arr) - 1 < 2 * i + 1:
            print(f"Adding left child {child} to parent {parent}")
            extend_arr_tree(self.arr, 2 * i + 1, child)
        elif len(self.arr) - 1 < 2 * i + 2:
            print(f"Adding right child {child} to parent {parent}")
            extend_arr_tree(self.arr, 2 * i + 2, child)
        else:
            assert False, f"Parent {parent} has already 2 children"

    def get_children(self, parent):
        assert parent in set(self.arr), f"{parent} is not present in the tree"
        i = self.arr.index(parent)
        ind1 = 2 * i + 1
        ind2 = 2 * i + 2
        children = []
        for ind in [ind1, ind2]:
            if ind < len(self.arr):
                if self.arr[ind] is not None:
                    children.append(self.arr[ind])
        return children

    def get_leaves(self, parent):
        p = parent
        leaves = []
        c = self.get_children(p)
        parents = collections.deque(c)
        while len(parents) > 0:
            p = parents.popleft()
            # print(f"Get children for parent {p}")
            c = self.get_children(p)
            if len(c) == 0:
                leaves.append(p)
            else:
                parents.extend(c)
        return leaves


if __name__ == '__main__':
    tree = Tree()
    tree.add_child(0, 'a')
    tree.add_child(0, 'b')
    tree.add_child('a', 'c')
    tree.add_child('b', 'd')
    tree.add_child('b', 'e')
    print(f"Tree array: {tree.arr}")
    print(f"Tree leaves: {tree.get_leaves(0)}")
