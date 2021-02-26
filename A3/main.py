# Closed Form Linear Regression
# Kevin Tayah
# CS383
import random, math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

EPSILON = 2 ** -23

np.random.seed(0) # seed can be inputed here

def load_data(file):
    # Load csv file
    data = np.genfromtxt(file, dtype=None, names=True, delimiter=',')
    np.random.shuffle(data)

    splits = np.array_split(data, 3)
    return np.concatenate((splits[0], splits[1])), splits[2]

def enumerate(data):
    _, enumerated = np.unique(data, return_inverse=True)
    return enumerated

def preprocess_1(data):
    p1 = np.array(data)
    p1['sex'] = enumerate(data['sex'])
    p1['smoker'] = enumerate(data['smoker'])
    p1['region'] = enumerate(data['region'])
    return p1

def preprocess_2(data):
    p2 = np.array(data)
    return p2

def preprocess_3(data):
    p3 = np.array(data)
    p3['sex'] = enumerate(data['sex'])
    p3['smoker'] = enumerate(data['smoker'])
    p3['region'] = enumerate(data['region'])
    return p3

def preprocess_4(data):
    p4 = np.array(data)
    return p4

def regression(X, Y):
    N = len(X)

    w = np.ones(X.shape[1]) # Weights
    b = 0 # Bias factor
    l = 0.001  # The learning Rate
    rmse = 10000

    while True:
        Y_pred = np.dot(X, w) + b
        D_w = (2 / N) * np.dot(X.T, (Y_pred - Y)) # ∂J/∂w
        D_b = sum(Y_pred - Y)/N # ∂J/∂b
        w = w - l * D_w
        b = b - l * D_b

        new_rmse = math.sqrt((1/N) * sum((Y_pred - Y)**2))
        print(rmse)
        if abs(rmse - new_rmse) < 0.1:
            rmse = new_rmse
            break
        else:
            rmse = new_rmse
    
    return w, b, new_rmse

def main():
    csv_file = 'insurance.csv'

    train, validation = load_data(csv_file)
    data = preprocess_1(train)
    X = np.array(data[['age', 'sex', 'bmi', 'children', 'smoker', 'region']].tolist()).astype(np.float)
    Y = np.array(data['charges'].tolist())
    
    weights, bias, rmse = regression(X, Y)

    print('weights:', weights)
    print('bias:', bias)
    print('rmse:', rmse)

if __name__ == '__main__':
    main()