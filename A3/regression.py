import math
import numpy as np

class Regression:
    STOPPING_FACTOR = 0.01 # can be adjusted for more or less percision for weights
    np.random.seed(0) # seed can be inputed here

    def __init__(self, X, Y, l = 0.001):
        self.X = X
        self.Y = Y
        self.l = l

        self.w = np.zeros(X.shape[1])
        self.b = 0
        self.rmse = 0

    def fit(self, useBias):
        N = len(self.X)

        while True:
            Y_pred = np.dot(self.X, self.w) + self.b
            D_w = (2 / N) * np.dot(self.X.T, (Y_pred - self.Y)) # ∂J/∂w
            self.w = self.w - self.l * D_w

            if useBias:
                D_b = (1/N) * np.sum(2 * (Y_pred - self.Y)) # ∂J/∂b
                self.b -= self.l * D_b

            new_rmse = math.sqrt((1/N) * sum((Y_pred - self.Y)**2))
            if abs(self.rmse - new_rmse) < self.STOPPING_FACTOR:
                self.rmse = new_rmse
                break
            else:
                self.rmse = new_rmse

        return self.w, self.b, self.rmse

    def predict(self, X):
        N = len(X)
        Y_pred = np.dot(X, self.w) + self.b
        rmse = math.sqrt((1/N) * sum((Y_pred - self.Y)**2))

        return Y_pred, rmse
        
    def getRmse(self):
        return self.rmse

    def getWeights(self):
        return self.w

    def getBias(self):
        return self.b