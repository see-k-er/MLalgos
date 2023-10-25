#Random Forest Python Implementation
#Pre-requisite is Decision Trees

#Imports
from DecisionTree import DecisionTree
import numpy as np
from collections import Counter

#Creating a random forest class
class RandomForest:

    #Initializing the rf parameters
    def __init__(self, n_trees=30, max_depth=15, min_samples_split=3, n_feature=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_feature
        self.trees = []

    #
    def fit(self,X,y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split,
                        n_features=self.n_features)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    #underlined are helper functions
    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True) #replacment sampling
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common


    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        #Output here is [[1,0,1,1], [0,0,1,1], []] -> prediction is a list sample wise
        #But we need the first element in the list to have the predicitons for sample 1
        #from all the trees and so on
        tree_preds = np.swapaxes(predictions, 0, 1)
        predicitons = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions
