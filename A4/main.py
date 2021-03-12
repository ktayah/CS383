# Assignment 4 - Naive Bayes & Logistic Regression
# Kevin Tayah
# CS383
import sys
import numpy as np
from logistic_regression import LogisticRegression
from naive_bayes import NaiveBayes

np.random.seed(0) # seed can be inputed here

def standardize(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0, ddof=1)

def load_data(file):
    # Load csv file
    data = np.genfromtxt(file, delimiter=',')
    np.random.shuffle(data)

    splits = np.array_split(data, 3)
    return np.concatenate((splits[0], splits[1])), splits[2]

def preprocess_data(data):
    return standardize(data[:,:-1]), data[:,-1]

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def naive_bayes():
    file = 'spambase.data'
    train, validation = load_data(file)

    # Standardize features and extract classes
    train_x, train_y = preprocess_data(train)
    validation_x, validation_y = preprocess_data(validation)

    # Run Naive Bayes fit and predict
    nb = NaiveBayes()
    nb.fit(train_x, train_y)
    predictions = nb.predict(validation_x)

    print('Accuracy:', accuracy(validation_y, predictions))

def logistic_regression():
    file = 'spambase.data'
    train, validation = load_data(file)

    # Standardize features and extract classes
    train_x, train_y = preprocess_data(train)
    validation_x, validation_y = preprocess_data(validation)

    # Run Logistic Regression fit and predict
    lr = LogisticRegression()
    lr.fit(train_x, train_y)
    predictions = lr.predict(validation_x)

    print('Accuracy:', accuracy(validation_y, predictions))

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'nb':
        naive_bayes()
    elif len(sys.argv) == 2 and sys.argv[1] == 'lr':
        logistic_regression()
    else:
        print('Incorrect usage. Execute "python main.py nb" for the naive bayes algorithm or "python main.py lr" for logistic regression algorithm')