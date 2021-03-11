# Logistic Regression Classifier
# Kevin Tayah
# CS383
import math
import numpy as np

np.random.seed(0) # seed can be inputed here

def standardize(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0, ddof=1)

def load_data(file):
    # Load csv file
    data = np.genfromtxt(file, delimiter=',')
    np.random.shuffle(data)

    splits = np.array_split(data, 3)
    return np.concatenate((splits[0], splits[1])), splits[2]

def main():
    file = 'spambase.data'
    train, validation = load_data(file)
    
    # Standardize features
    stdTrain = standardize(train[:,:-1])
    stdValidation = standardize(validation[:,:-1])

if __name__ == '__main__':
    main()