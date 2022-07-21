

#just Run this .py file


import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import time




class Node:
    def __init__(self,feature=None,threshold=None,left=None,right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTree:

    def __init__(self, max_depth=100, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def IsFinished(self, depth):
        if (depth >= self.max_depth or self.n_class_labels == 1
                or self.n_samples < self.min_samples_split):
            return True
        return False

    def FIT(self, X, y):
        self.root = self.BuildTree(X, y)

    def calculateEntropy(self, y):
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return entropy

    def split(self, X, thresh):
        left_idx = np.argwhere(X <= thresh).flatten()
        right_idx = np.argwhere(X > thresh).flatten()
        return left_idx, right_idx

    def InformationGain(self, X, y, thresh):
        parent_loss = self.calculateEntropy(y)
        left_idx, right_idx = self.split(X, thresh)
        n, n_left, n_right = len(y), len(left_idx), len(right_idx)
        if n_left == 0 or n_right == 0:
            return 0
        child_loss = (n_left / n) * self.calculateEntropy(
            y[left_idx]) + (n_right / n) * self.calculateEntropy(y[right_idx])
        return parent_loss - child_loss

    def BestSplit(self, X, y, features):
        split = {'score': -1, 'feat': None, 'thresh': None}
        for feat in features:
            X_feat = X[:, feat]
            thresholds = np.unique(X_feat)
            for thresh in thresholds:
                score = self.InformationGain(X_feat, y, thresh)
                if score > split['score']:
                    split['score'] = score
                    split['feat'] = feat
                    split['thresh'] = thresh
        return split['feat'], split['thresh']

    def BuildTree(self, X, y, depth=0):
        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))
        rnd_feats = np.random.choice(self.n_features,
                                     self.n_features,
                                     replace=False)
        best_feat, best_thresh = self.BestSplit(X, y, rnd_feats)
        if self.IsFinished(depth):
            most_common_Label = np.argmax(np.bincount(y))
            return Node(value=most_common_Label)
        left_idx, right_idx = self.split(X[:, best_feat], best_thresh)
        left_child = self.BuildTree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self.BuildTree(X[right_idx, :], y[right_idx],
                                       depth + 1)
        return Node(best_feat, best_thresh, left_child, right_child)

    def traverse(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self.traverse(x, node.left)
        return self.traverse(x, node.right)

    def predict(self, X):
        predictions = [self.traverse(x, self.root) for x in X]
        return np.array(predictions)


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy



df = pd.read_csv(r'C:\Users\Abdelrahman Muhsen\Desktop\task3_AI\cardio_train.csv',sep=";")
X = np.array(df[df.columns[1:-1]])
Y = np.array(df.cardio)
X_train, X_test, y_train, y_test = train_test_split(X,Y)
Dtree = DecisionTree(max_depth=10)
start = time.time()
Dtree.FIT(X_train, y_train)
end= time.time()
print("training time = ", end - start)
start = time.time()
y_pred1 = Dtree.predict(X_test)
end = time.time()
print("test time = ", end - start)
Dtree_accuracy = np.round(accuracy(y_test, y_pred1),4)
start = time.time()
Skmodel = tree.DecisionTreeClassifier(criterion="entropy", max_depth=10)
Skmodel.fit(X_train, y_train)
end = time.time()
print("training time of SKmodel = ",  end - start)
start = time.time()
y_pred_SK = Skmodel.predict(X_test)
end = time.time()
print("test tome of SKmodel = ",  end - start)
SKmodel_accuracy = np.round(accuracy(y_test, y_pred_SK),4)
print("Dtree Accuracy:", Dtree_accuracy, "SKmodel accuracy2: ", SKmodel_accuracy)