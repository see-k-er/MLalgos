#Implementing KNN algorithm from scratch

#imports
import numpy as np
from collections import Counter

#euclidean distance global function
def euclidean_distance(x1, x2):
    edistance = np.sqrt(np.sum((x2-x1)**2))
    return edistance


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    #to calculate the distance from the given point to every other point in the dataset
    def predict(self, X):
        predictions = [ self._predict(x) for x in X]
        return predictions

    #helper function for main predict function
    def _predict(self, x):
        #compute the distances -> euclidean distance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        #get the closest K
        k_indices = np.argsort(distances)[:self.k] #argsort returns the indices after sorting
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        #determine label with majority vote
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
