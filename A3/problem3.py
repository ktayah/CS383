# Closed Form Linear Regression
# Kevin Tayah
# CS383
import random, math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

STOPPING_FACTOR = 0.01 # can be adjusted for more or less percision for weights
np.random.seed(0) # seed can be inputed here

def load_data(file):
    # Load csv file
    data = np.genfromtxt(file, dtype=None, names=True, delimiter=',')
    np.random.shuffle(data)

    splits = np.array_split(data, 3)
    return np.concatenate((splits[0], splits[1])), splits[2]

def enumerateData(data):
    _, enumerated = np.unique(data, return_inverse=True)
    return enumerated

def preprocess_1(data):
    useBias = False
    p1 = np.array(data)

    p1['sex'] = enumerateData(data['sex'])
    p1['smoker'] = enumerateData(data['smoker'])
    p1['region'] = enumerateData(data['region'])
    return p1, useBias

def preprocess_2(data):
    useBias = True
    p2 = np.array(data)

    p2['sex'] = enumerateData(data['sex'])
    p2['smoker'] = enumerateData(data['smoker'])
    p2['region'] = enumerateData(data['region'])
    return p2, useBias

def preprocess_3(data):
    useBias = False
    N = len(data)

    enumeratedSex = enumerateData(data['sex'])
    enumeratedSmoker = enumerateData(data['smoker'])
    enumeratedRegions = enumerateData(data['region'])

    p3 = np.array([
        data['age'],
        enumeratedSex,
        data['bmi'],
        data['children'],
        enumeratedSmoker,
        np.zeros(N),
        np.zeros(N),
        np.zeros(N),
        np.zeros(N),
        data['charges']
    ]).T

    for i, enumeratedRegion in enumerate(enumeratedRegions):
        p3[i, 5 + enumeratedRegion] = 1

    return p3, useBias

def preprocess_4(data):
    useBias = True
    N = len(data)

    enumeratedSex = enumerateData(data['sex'])
    enumeratedSmoker = enumerateData(data['smoker'])
    enumeratedRegions = enumerateData(data['region'])

    p4 = np.array([
        data['age'],
        enumeratedSex,
        data['bmi'],
        data['children'],
        enumeratedSmoker,
        np.zeros(N),
        np.zeros(N),
        np.zeros(N),
        np.zeros(N),
        data['charges']
    ]).T

    for i, enumeratedRegion in enumerate(enumeratedRegions):
        p4[i, 5 + enumeratedRegion] = 1

    return p4, useBias

def validate(X, Y, w, b):
    N = len(X)
    Y_pred = np.dot(X, w) + b
    rmse = math.sqrt((1/N) * sum((Y_pred - Y)**2))

    return Y_pred, rmse

def regression(X, Y, useBias):
    N = len(X)

    w = np.zeros(X.shape[1]) # weights
    b = 0 # bias factor
    l = 0.0001  # learning Rate
    rmse = 0

    while True:
        Y_pred = np.dot(X, w) + b
        D_w = (2 / N) * np.dot(X.T, (Y_pred - Y)) # ∂J/∂w
        w -= l * D_w

        if useBias:
            D_b = (1/N) * np.sum(2 * (Y_pred - Y)) # ∂J/∂b
            b -= l * D_b

        new_rmse = math.sqrt((1/N) * sum((Y_pred - Y)**2))
        if abs(rmse - new_rmse) < STOPPING_FACTOR:
            rmse = new_rmse
            break
        else:
            rmse = new_rmse
    
    return w, b, new_rmse

def main():
    csv_file = 'insurance.csv'

    train, validation = load_data(csv_file)
    
    data_train, useBias = preprocess_4(train)
    data_validate, useBias = preprocess_4(validation)

    # Use this for preprocessing 1 or 2
    # X_train = np.array(data_train[['age', 'sex', 'bmi', 'children', 'smoker', 'region']].tolist()).astype(np.float)
    # Y_train = np.array(data_train['charges'].tolist())
    # X_validate = np.array(data_validate[['age', 'sex', 'bmi', 'children', 'smoker', 'region']].tolist()).astype(np.float)
    # Y_validate = np.array(data_validate['charges'].tolist())

    # Use this for preprocessing 3 or 4
    X_train = data_train[:len(data_train), :len(data_train[0]) - 1]
    Y_train = data_train[:len(data_train), len(data_train[0]) - 1]
    X_validate = data_validate[:len(data_validate), :len(data_validate[0]) - 1]
    Y_validate = data_validate[:len(data_validate), len(data_validate[0]) - 1]
    
    weights, bias, rmse_train = regression(X_train, Y_train, useBias)

    print('Training information')
    print('weights:', weights)
    print('bias:', bias)
    print('rmse:', rmse_train)

    Y_pred, rmse_validate = validate(X_validate, Y_validate, weights, bias)

    print('Validation info')
    print('rmse:', rmse_validate)

if __name__ == '__main__':
    main()