# Logistic Regression Classifier
# Kevin Tayah
# CS383
import numpy as np

EPOCHES = 1000
LEARNING_RATE = 0.0001

class LogisticRegression:
    def __init__(self, l = LEARNING_RATE, epoches = EPOCHES):
        self.l = l
        self.epoches = epoches
        self.weights = None

    def _sigmoid(_, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, X):
        Y_pred = self._sigmoid(np.dot(X, self.weights))
        Y_pred_classes = [1 if i > 0.5 else 0 for i in Y_pred]
        return Y_pred_classes
    
    def fit(self, X, y):
        _, n_features = X.shape
        self.weights = np.zeros(n_features)

        for _ in range(self.epoches):
            Y_pred = self._sigmoid(np.dot(X, self.weights))
            D_w = np.dot(X.T, (y - Y_pred))
            self.weights += self.l * D_w
