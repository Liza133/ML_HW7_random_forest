from node import Node
import numpy as np


class Tree:
    def __init__(self):
        self.nodes = []

    def get_tree(self):
        s = []
        for node in self.nodes:
            s.append(node.get_node())
        return sorted(s, key=lambda n: n[0])

    def grow_branches(self, node):
        return self.nodes.append(node)

    def get_classes(self, inputs):
        classes = []
        for x in inputs:
            root = self.nodes[0]
            term = False
            while not term:
                if root.T == None:
                    if x[root.split_ind] < root.split_val:
                        root = root.left
                    else:
                        root = root.right
                else:
                    classes.append(np.argmax(root.T))
                    term = True
        return np.array(classes)


class DT:
    def __init__(self, max_depth, min_entropy, min_elem):
        self.max_depth = max_depth
        self.min_entropy = min_entropy
        self.min_elem = min_elem
        self.root = Node()

    def count(self, mas, val):
        count = 0
        for el in mas:
            if el == val:
                count += 1
        return count

    def calculate_entropy(self, targets):
        entropy = 0
        for i in range(10):
            c = self.count(targets, i) / len(targets)
            if c != 0:
                entropy += -c * np.log(c)
        return entropy

    def terminal_node_output(self, targets):
        return [self.count(targets, i) / len(targets) for i in range(10)]

    def information_gain(self, entropy_val, targets_left, targets_right):
        H_i0 = self.calculate_entropy(targets_left)
        H_i1 = self.calculate_entropy(targets_right)
        N = len(targets_left) + len(targets_right)
        return entropy_val - (len(targets_left) * H_i0 + len(targets_right) * H_i1) / N

    def split_set(self, psi_input, split_val):
        left_ind = []
        right_ind = []
        for i in range(len(psi_input)):
            if psi_input[i] > split_val:
                right_ind.append(i)
            else:
                left_ind.append(i)
        return left_ind, right_ind

    def buildTree(self, inputs, targets, node, depth, tree, psi, thau):
        entropy_val = self.calculate_entropy(targets)
        if (depth >= self.max_depth or entropy_val <= self.min_entropy or len(targets) <= self.min_elem):
            node.T = self.terminal_node_output(targets)
            node.depth = depth
            tree.grow_branches(node)
        else:
            I = []
            split_ind = []
            split_val = []
            inputs_left = []
            targets_left = []
            inputs_right = []
            targets_right = []
            for i in psi:
                for t in thau:
                    l_r_ind = self.split_set(inputs[:, i], t)
                    if len(l_r_ind[0]) != 0 and len(l_r_ind[1]) != 0:
                        split_val.append(t)
                        inputs_left.append(inputs[l_r_ind[0]])
                        inputs_right.append(inputs[l_r_ind[1]])
                        targets_left.append(targets[l_r_ind[0]])
                        targets_right.append(targets[l_r_ind[1]])
                        I.append(self.information_gain(entropy_val, targets_left[-1], targets_right[-1]))
                        split_ind.append(i)
            argmaxI = np.argmax(I)
            node.split_val = split_val[argmaxI]
            node.split_ind = split_ind[argmaxI]
            node.left = Node()
            node.right = Node()
            node.depth = depth
            tree.grow_branches(node)
            self.buildTree(inputs_left[argmaxI], targets_left[argmaxI], node.left, depth + 1, tree, psi, thau)
            self.buildTree(inputs_right[argmaxI], targets_right[argmaxI], node.right, depth + 1, tree, psi, thau)
