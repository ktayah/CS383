# Naive Bayes Classifier
# Kevin Tayah
# CS383
import numpy as np

EPSILON = 2**-23

class NaiveBayes:
    def _gaussian_probability(self, x, c_index):
        mean = self.mean[c_index]
        var = self.var[c_index]

        return np.exp(-(x - mean)**2 / (2 * var**2)) / (var * np.sqrt(2 * np.pi))

    def _calculate_nb(self, x):
        posteriors = []

        for c in self.classes:
            c_index = int(c)
            prior = np.log(self.priors[c_index])
            prob = self._gaussian_probability(x, c_index) + EPSILON # Add a small value to avoid log(0)
            class_conditionals = np.sum(np.log(prob))

            posterior = prior + class_conditionals
            posteriors.append(posterior)
        
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        prediction_y = [self._calculate_nb(x) for x in X]
        return prediction_y

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # initialize mean, variance, and priors for our features
        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        # set mean, variance, and priors for each of our features
        for c in self.classes:
            c_index = int(c)
            X_c = X[c_index==y] # grab samples of all the same classes
            X_c_n = X_c.shape[0]

            self.mean[c_index,:] = X_c.mean(axis=0)
            self.var[c_index,:] = X_c.var(axis=0, ddof=1)
            self.priors[c_index] = X_c_n / float(n_samples)
