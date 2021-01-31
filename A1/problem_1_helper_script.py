import numpy as np

X = [0, 0, 1, 0, 1, 1, 1, 1, 2, 2], [1, 0, 1, 0, 1, 0, 0, 1, 0, 1]
Y = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

x0mean = np.mean(X[0])
x0std = np.std(X[0], ddof=1)

x1mean = np.mean(X[1])
x1std = np.std(X[1], ddof=1)

standardX0 = X[0] - x0mean / x0std
standardX1 = X[1] - x1mean / x1std

array = np.array([standardX0, standardX1]).T

covariance_matrix = np.cov(array.T, ddof=1)
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

first_pca = np.dot(array, eigen_vectors[:,0])
second_pca = np.dot(array, eigen_vectors[:,1])

print(first_pca)
print(second_pca)

