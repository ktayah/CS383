# Clustering
# Kevin Tayah
# CS383
import random
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

colors = ['blue', 'red', 'orange', 'green', 'cyan', 'purple', 'black']
MAX_PCA = 3
EPSILON = 2 ** -23

random.seed(100) # seed can be inputed here

def load_xy(file):
    data = np.genfromtxt(file, delimiter=',')
    y = data[:,0]
    x = data[:,1:]
    return x, y

def standardize(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0, ddof=1)

def my_K_means(x, y, k):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if len(x[0]) > MAX_PCA:
        x = pca(x, MAX_PCA).T

    # This grabs a random k amount of observations to be our k_vectors
    r_k_indices = random.sample(range(0, x.shape[0]), k)
    k_vectors = x[r_k_indices,:]

    iteration = 0
    while True:
        print('Iteration:', iteration)
        iteration += 1
        ax.clear()

        for i, k_vector in enumerate(k_vectors):
            # This labels the k_vector in the scatter plot as a red circle
            ax.scatter(k_vector[0], k_vector[1], zs=k_vector[2], s=40, c=colors[i])
        
        # Initialize clusters dictionary
        clusters = {}
        for i in range(0, k):
            clusters[i] = []

        for observation in x:
            distances = []

            for k_vector in k_vectors:
                distances.append(np.linalg.norm(observation - k_vector))

            min_k_index = np.array(distances).argmin()
            clusters[min_k_index].append(observation)

        for i, cluster in enumerate(clusters.values()):
            # Set the scatterplot
            _cluster = np.array(cluster)
            _x = _cluster[:,0]
            _y = _cluster[:,1]
            _z = _cluster[:,2]
            ax.scatter(_x, _y, zs=_z, c=colors[i], marker='x')

        new_k_vectors = []
        sum_l1_distance = 0
        for index, cluster in enumerate(clusters.values()):
            # Set the new k_vectors
            new_k_vector = np.mean(cluster, axis=0)
            new_k_vectors.append(new_k_vector)
            sum_l1_distance += np.linalg.norm((k_vectors[index] - new_k_vector), ord=1)

        if sum_l1_distance <= EPSILON:
            break
        else:
            k_vectors = new_k_vectors
        
    plt.show()


def pca(data, n=2):
    covariance_matrix = np.cov(data.T, ddof=1)
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

    max_eigen_value_indices = (-eigen_values).argsort()[:n] # Finds indices of n highest eigen_values

    pca = []
    for max_eigen_value_index in max_eigen_value_indices:
        pca.append(np.dot(data, eigen_vectors[:,max_eigen_value_index]))

    return np.array(pca)

def main():
    csv_file = 'diabetes.csv'
    clusters = 2

    x, y = load_xy(csv_file)
    std_x = standardize(x)

    my_K_means(std_x, y, clusters)
    return

if __name__ == '__main__':
    main()