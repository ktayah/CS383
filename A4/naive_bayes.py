# Naive Bayes Classifier
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

def create_gaussian(data):
    _, D = data.shape
    gaussian = np.zeros(data.shape)

    for i in range(D):
        feature = data[:,i]
        featureMean = np.mean(feature)
        featureStd = np.std(feature, ddof=1)

        spamGaussian = (1 / (featureStd * math.sqrt(2 * math.pi))) * math.e ** (-((feature - featureMean)**2) / (2 * featureStd**2))
        gaussian[:,i] = spamGaussian
    
    return gaussian

def main():
    file = 'spambase.data'
    train, validation = load_data(file)

    # Standardize features and extract classes
    stdTrain = standardize(train[:,:-1])
    stdValidation = standardize(validation[:,:-1])
    # trainClasses = train[:,-1:]
    # validationClasses = train[:,-1:]

    # Divide training data into Positive and Negative Samples
    stdTrainSpam = stdTrain[train[:, len(train[0]) - 1] == 1]
    stdTrainNotSpam = stdTrain[train[:, len(train[0]) - 1] == 0]

    # Create Guassian Models
    spamGaussian = create_gaussian(stdTrainSpam)
    notSpamGaussian = create_gaussian(stdTrainNotSpam)

    print(spamGaussian, notSpamGaussian)

if __name__ == '__main__':
    main()