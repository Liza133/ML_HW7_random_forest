class Node:
    def __init__(self):
        self.right = None
        self.left = None
        self.split_ind = None
        self.split_val = None
        self.depth = None
        self.T = None


    def get_node(self):
        if self.T != None:
            return self.depth, self.T
        else:
            return self.depth, self.split_ind, self.split_val