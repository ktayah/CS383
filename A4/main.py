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

def precision(y_true, y_pred):
    true_positives = np.sum(y_true == y_pred)
    f_p_bools = [y_true[i] != y_p and y_p == 1 for i, y_p in enumerate(y_pred)]
    false_positives = np.sum(f_p_bools)
    return true_positives / true_positives + false_positives

def recall(y_true, y_pred):
    true_positives = np.sum(y_true == y_pred)
    f_n_bools = [y_true[i] != y_p and y_p == 0 for i, y_p in enumerate(y_pred)]
    false_negatives = np.sum(f_n_bools)
    return true_positives / true_positives + false_negatives

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def f_measure(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (2 * p  * r) / (p + r)

def main():
    file = 'spambase.data'
    train, validation = load_data(file)

    # Standardize features and extract classes
    train_x, train_y = preprocess_data(train)
    validation_x, validation_y = preprocess_data(validation)

    predictions = None
    if sys.argv[1] == 'nb':
        # Run Naive Bayes fit and predict
        nb = NaiveBayes()
        nb.fit(train_x, train_y)
        predictions = nb.predict(validation_x)
    else:
        # Run Logistic Regression fit and predict
        lr = LogisticRegression()
        lr.fit(train_x, train_y)
        predictions = lr.predict(validation_x)

    print('Precision:', precision(validation_y, predictions))
    print('Recall:', recall(validation_y, predictions))
    print('f-measure:', f_measure(validation_y, predictions))
    print('Accuracy:', accuracy(validation_y, predictions))

if __name__ == '__main__':
    if len(sys.argv) == 2:
        main()
    else:
        print('Incorrect usage. Execute "python main.py nb" for the naive bayes algorithm or "python main.py lr" for logistic regression algorithm')