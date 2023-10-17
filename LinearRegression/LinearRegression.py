#Code to implement Linear Regression from scratch without scikit learn
import numpy as np

class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        #initializing all the parameters required
          self.lr = lr
          self.n_iters = n_iters
          self.weights = None
          self.bias = None

    #Fitting the dataset to the model
    def fit(self,X,y):
        #Initializing the weights and biases
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        #Repeat the steps for the set number of iterations
        for _ in range(self.n_iters):
            #Finding y_pred = wX + b
            y_pred = np.dot(X, self.weights) + self.bias

            #Finding the derivatives dw and db:
            dw = (1/n_samples) * 2 * np.dot(X.T,(y_pred-y))
            db = (1/n_samples) * 2 * np.sum(y_pred-y)

            #Updating the weights and bias
            self.weights = self.weights - (self.lr * dw)
            self.bias = self.bias - (self.lr * db)

    #Predicting the values for given dataset
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
