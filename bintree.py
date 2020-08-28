#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-06-26 09:08:41 (UTC+0200)

import copy
import numpy
import scipy.optimize
import collections


def extend_arr_tree(arr, index, val):
    """
    Add given val at given ind in arr
    """
    assert index > len(arr) - 1
    arr.extend([None, ] * (index - len(arr)))
    arr.append(val)


class Tree(object):
    def __init__(self, verbose=False):
        # Array data structure for the tree
        # See: https://en.wikipedia.org/w/index.php?title=Binary_tree&oldid=964115444#Arrays
        self.arr = [0, ]
        self.verbose = verbose

    @property
    def depth(self):
        return numpy.int(numpy.ceil(numpy.log(1 - (1 - 2) * len(self.arr)) / numpy.log(2))) - 1

    @property
    def nodes(self):
        return [e for e in self.arr if e is not None]

    def linkage_to_tree(self, linkage_matrix):
        """
        Convert a linkage matrix to an array data structure
        """
        n = linkage_matrix.shape[0] + 1
        parents = [2 * (n - 1), ]
        self.arr = list(numpy.copy(parents))
        leaves = range(n)
        i = 0
        new_parents = []
        while True:
            while i < len(parents):
                print(i, parents)
                parent = parents[i]
                children = numpy.int_(linkage_matrix[int((parent) % n)])
                self.add_child(parent, children[0])
                self.add_child(parent, children[1])
                new_parents.extend(list(set(children) - set(leaves)))
                i += 1
            parents = list(set(new_parents) - set(leaves))
            new_parents = []
            i = 0
            if len(parents) == 0:
                break
        self.close()

    def add_child(self, parent, child):
        assert parent in set(self.arr), f"{parent} is not present in the tree"
        assert child not in set(self.arr), f"{child} already in tree"
        i = self.arr.index(parent)
        if len(self.arr) - 1 < 2 * i + 1:
            if self.verbose:
                print(f"Adding left child {child} to parent {parent}")
            extend_arr_tree(self.arr, 2 * i + 1, child)
        elif len(self.arr) - 1 < 2 * i + 2:
            if self.verbose:
                print(f"Adding right child {child} to parent {parent}")
            extend_arr_tree(self.arr, 2 * i + 2, child)
        else:
            if self.arr[2 * i + 1] is None:
                self.arr[2 * i + 1] = child
            elif self.arr[2 * i + 2] is None:
                self.arr[2 * i + 2] = child
            else:
                assert False, f"Parent {parent} has already 2 children"

    def close(self):
        """
        Close the tree building step procedure
        """
        arr_len = (1 - 2**(self.depth + 1)) / (1 - 2)
        toadd = int(arr_len - len(self.arr))
        self.arr.extend([None, ] * toadd)

    def get_children(self, parent, return_index=False):
        assert parent in set(self.arr), f"{parent} is not present in the tree"
        i = self.arr.index(parent)
        ind1 = 2 * i + 1
        ind2 = 2 * i + 2
        children = []
        inds = []
        for ind in [ind1, ind2]:
            if ind < len(self.arr):
                if self.arr[ind] is not None:
                    children.append(self.arr[ind])
                    inds.append(ind)
        if return_index:
            return children, inds
        else:
            return children

    def get_parent(self, child):
        assert child in set(self.arr), f"{child} is not present in the tree"
        i = self.arr.index(child)
        ind = int((i - 1) / 2)
        return self.arr[ind]

    def get_leaves(self, parent, return_offspring=False,
                   return_offspring_index=False):
        p = parent
        leaves = []
        offspring = []
        offspring_inds = []
        c, inds = self.get_children(p, return_index=True)
        offspring.extend(c)
        offspring_inds.extend(inds)
        parents = collections.deque(c)
        while len(parents) > 0:
            p = parents.popleft()
            c, inds = self.get_children(p, return_index=True)
            offspring.extend(c)
            offspring_inds.extend(inds)
            if len(c) == 0:
                leaves.append(p)
            else:
                parents.extend(c)
        if return_offspring:
            if return_offspring_index:
                return leaves, offspring, offspring_inds
            else:
                return leaves, offspring
        else:
            return leaves

    def get_subtree(self, node):
        """
        Returns the index of the nodes of the subtree rooted on node
        """
        i_list = collections.deque([self.arr.index(node), ])
        i = i_list.popleft()
        inds = [i]
        while (2 * i + 2) < len(self.arr):
            inds_ = [2 * i + 1, 2 * i + 2]
            inds.extend(inds_)
            i_list.extend(inds_)
            i = i_list.popleft()
        return inds

    def print_tree(self, doprint=True):
        ind2 = 1
        linewidth = 2**self.depth * 2
        lines = []
        for i in range(self.depth + 1):
            ind1, ind2 = ind2, 2 * ind2
            line = self.arr[ind1 - 1:ind2 - 1]
            nchars = len(line)
            spacewidth = int(linewidth / (2 * nchars))
            spacestr = ' ' * spacewidth
            line = spacestr.join([str(e) if e is not None else '-' for e in line])
            line = line.center(linewidth)
            lines.append(line)
        lines = '\n'.join(lines)
        if doprint:
            print(lines)
        return lines

    def get_nodes(self, depth):
        """
        Get the nodes at the given tree depth
        """
        ind1 = 2**(depth) - 1
        ind2 = 2**(depth + 1) - 1
        return [e for e in self.arr[ind1:ind2] if e is not None]

    def get_node_depth(self, node):
        """
        Return the depth of the node
        """
        ind = self.arr.index(node)
        return numpy.int(numpy.ceil(numpy.log(1 - (1 - 2) * (ind + 1)) / numpy.log(2))) - 1

    def swap_branches(self, node1, node2):
        assert self.get_parent(node1) == self.get_parent(node2), f"To swap 2 branches, nodes must have the same parents. {node1} has as parent {self.get_parent(node1)} and {node2} has as parent {self.get_parent(node2)}"
        if self.verbose:
            print(f"Swapping branches {node1} and {node2}")
        inds1 = self.get_subtree(node1)
        inds2 = self.get_subtree(node2)
        arr = numpy.asarray(copy.deepcopy(self.arr))
        i = min(min(inds1), min(inds2))
        arr[:i] = self.arr[:i]
        arr[inds1] = numpy.asarray(self.arr)[inds2]
        arr[inds2] = numpy.asarray(self.arr)[inds1]
        arr = list(arr)
        self.arr = arr

    def __repr__(self):
        return self.print_tree(doprint=False)


class Align(object):
    """
    Align 2 trees
    """
    def __init__(self, tree1, tree2):
        self.tree1 = tree1
        self.tree2 = tree2
        self.nodes1 = self.tree1.nodes
        self.nodes2 = self.tree2.nodes
        self.overlaps = self.get_overlaps()

    def get_overlaps(self, gap=1):
        """
        Get the overlap matrix for each depth level
        """
        overlaps = {}
        for node1 in self.nodes1:
            leaves1 = self.tree1.get_leaves(node1)
            for node2 in self.nodes2:
                depth1 = tree1.get_node_depth(node1)
                depth2 = tree1.get_node_depth(node2)
                leaves2 = self.tree2.get_leaves(node2)
                overlap = len((set(leaves1) | set([node1])) & (set(leaves2) | set([node2])))
                # Gap penalty:
                overlap -= numpy.abs(depth2 - depth1) * gap
                if node1 not in overlaps:
                    overlaps[node1] = {node2: overlap}
                else:
                    overlaps[node1][node2] = overlap
        return overlaps

    def _align(self, depth):
        nodes1 = self.tree1.get_nodes(depth)
        nodes2 = self.tree2.get_nodes(depth)
        score = 0
        for node1 in nodes1:
            for node2 in nodes2:
                s = self.overlaps[node1][node2]
                if s >= score:
                    node2_0 = self.tree2.arr[self.tree1.arr.index(node1)]
                    swapped = False
                    try:
                        self.tree2.swap_branches(node2_0, node2)
                        swapped = True
                    except AssertionError:
                        pass
                    if swapped:
                        score = s

    def align(self):
        depth1 = self.tree1.depth
        depth2 = self.tree2.depth
        assert depth1 == depth2
        depth = depth1
        for d in range(depth + 1):
            self._align(d)

    @property
    def score(self):
        arr1 = numpy.asarray(self.tree1.arr)
        arr2 = numpy.asarray(self.tree2.arr)
        leaves1 = numpy.asarray(self.tree1.get_leaves(0))
        leaves2 = numpy.asarray(self.tree2.get_leaves(0))
        s1 = numpy.isin(arr1, leaves1)
        s2 = numpy.isin(arr2, leaves2)
        isleaf = numpy.logical_and(s1, s2)
        overlap = (arr1 == arr2)
        return numpy.logical_and(isleaf, overlap).sum()


if __name__ == '__main__':
    tree = Tree(verbose=True)
    tree.add_child(0, 'a')
    tree.add_child(0, 'b')
    tree.add_child('a', 'c')
    tree.add_child('b', 'd')
    tree.add_child('b', 'e')
    tree.add_child('e', 'f')
    tree.add_child('e', 'g')
    tree.close()
    print(tree)
    print(f"Tree array: {tree.arr}")
    print(f"Tree depth: {tree.depth}")
    print(f"Tree leaves: {tree.get_leaves(0)}")
    print(f"Depth of each node: {['node '+str(n)+': '+str(tree.get_node_depth(n)) for n in tree.nodes]}")
    depth = 2
    print(f"Get nodes at depth {depth}: {tree.get_nodes(depth)}")
    leaves, offspring, offspring_inds = tree.get_leaves('b',
                                                        return_offspring=True,
                                                        return_offspring_index=True)
    print(f"Parent of nodes 'd' and 'e': {tree.get_parent('d'), tree.get_parent('e')}")
    print(f"Leaves from node 'b': {leaves}")
    print(f"Offspring from node 'b': {offspring}")
    print(f"Offspring indices from node 'b': {offspring_inds}")
    print("================================================================================")
    tree.swap_branches('a', 'b')
    print(tree)
    tree.swap_branches('b', 'a')
    print(tree)
    print("================================================================================")
    print("Aligning trees")
    tree1 = Tree()
    tree1.add_child(0, 1)
    tree1.add_child(0, 2)
    tree1.add_child(1, 'a')
    tree1.add_child(2, 3)
    tree1.add_child(2, 4)
    tree1.add_child(3, 'c')
    tree1.add_child(3, 'd')
    tree1.add_child(4, 'e')
    tree1.add_child(4, 'f')
    tree1.close()
    print(f"Tree 1:\n{tree1}")
    tree2 = Tree()
    tree2.add_child(0, 1)
    tree2.add_child(0, 2)
    tree2.add_child(1, 3)
    tree2.add_child(1, 4)
    tree2.add_child(2, 'a')
    tree2.add_child(3, 'f')
    tree2.add_child(3, 'e')
    tree2.add_child(4, 'd')
    tree2.add_child(4, 'c')
    tree2.close()
    print(f"Tree 2:\n{tree2}")
    align = Align(tree1, tree2)
    align.align()
    print(f"Alignment score: {align.score}")
    print(f"Tree 2 aligned on Tree 1:\n{align.tree2}")
    print("--------------------------------------------")
    print("Aligning new Tree 2 on Tree 1")
    tree2 = Tree()
    tree2.add_child(0, 1)
    tree2.add_child(0, 2)
    tree2.add_child(1, 3)
    tree2.add_child(1, 4)
    tree2.add_child(2, 'a')
    tree2.add_child(3, 'c')
    tree2.add_child(3, 'e')
    tree2.add_child(4, 'd')
    tree2.add_child(4, 'f')
    tree2.close()
    print(f"Tree 2:\n{tree2}")
    align = Align(tree1, tree2)
    align.align()
    print(f"Alignment score: {align.score}")
    print(f"Tree 2 aligned on Tree 1:\n{align.tree2}")
    print("--------------------------------------------")
    print("Building tree from a linkage matrix")
    linkage_mat = numpy.asarray([[11., 12.]
                                 , [0., 4.]
                                 , [8., 9.]
                                 , [6., 10.]
                                 , [1., 5.]
                                 , [2., 14.]
                                 , [16., 17.]
                                 , [7., 15.]
                                 , [3., 18.]
                                 , [19., 20.]
                                 , [13., 22.]
                                 , [21., 23.]])
    tree = Tree(verbose=True)
    tree.linkage_to_tree(linkage_mat)
