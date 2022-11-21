from tree import DT, Tree
from node import Node
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import plotly.graph_objs as go


def accuracy(val, tar):
    true_val = 0
    for i in range(len(val)):
        if val[i] == tar[i]:
            true_val += 1
    return true_val / len(val)


digits = load_digits()
X = digits.data
target = digits.target

# reshaping for OneHotEncoder
integer_encoded_reshape = target.reshape(len(target), 1)

# One-Hot Encoding
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded_reshape)

# Normalisation
for i in range(X.shape[0]):
    X[i] = 2 * (X[i] - np.amin(X[i])) / (np.amax(X[i]) - np.amin(X[i])) - 1

# Shuffle data
tr = 0.8
val = 0.1
N = X.shape[0]
ind_prm = np.random.permutation(np.arange(N))
train_ind = ind_prm[:int(tr * N)]
valid_ind = ind_prm[int(tr * N):int((val + tr) * N)]
test_ind = ind_prm[int((val + tr) * N):]
X_train, target_train, T_train = X[train_ind], target[train_ind], onehot_encoded[train_ind]
X_valid, target_valid, T_valid = X[valid_ind], target[valid_ind], onehot_encoded[valid_ind]
X_test, target_test, T_test = X[test_ind], target[test_ind], onehot_encoded[test_ind]


def get_rand_tree():
    L_1 = np.random.randint(10, 40)
    L_2 = np.random.randint(5, 35)
    ps = np.arange(0, 64, 1)
    psi = np.random.choice(ps, L_1, replace=False)
    th = np.arange(-1.0, 1.0, 0.02)
    thau = np.random.choice(th, L_2, replace=False)
    tree = Tree()
    root = Node()
    dt = DT(10, 0.01, 10)
    dt.buildTree(X_train, target_train, root, 1, tree, psi, thau)
    # r = tree.get_classes(X_valid)
    return tree


M = np.random.randint(10, 20)
trees = []
accuracies = []
for i in range(M):
    trees.append(get_rand_tree())
    accuracies.append(accuracy(trees[-1].get_classes(X_valid), target_valid))
max_accur_ind = np.argsort(accuracies)[-10:]
for i in max_accur_ind:
    print(trees[i].get_tree())
    print(accuracies[i])

