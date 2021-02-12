# Clustering Algorithm
# Kevin Tayah
# CS383
import random, sys
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

colors = ['blue', 'red', 'orange', 'green', 'cyan', 'purple', 'black']
MAX_PCA = 3
EPSILON = 2 ** -23

random.seed(0) # seed can be inputed here

def load_xy(file):
    # Load csv file
    data = np.genfromtxt(file, delimiter=',')
    y = data[:,0]
    x = data[:,1:]
    return x, y

def standardize(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0, ddof=1)

def my_K_means(x, y, k):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    video = cv2.VideoWriter('K_{}.avi'.format(str(k)), cv2.VideoWriter_fourcc(*'mp4v'), 1, (640, 480))

    if len(x[0]) > MAX_PCA:
        x = pca(x, MAX_PCA).T

    # This grabs a random k amount of observations to be our k_vectors
    r_k_indices = random.sample(range(0, x.shape[0]), k)
    k_vectors = x[r_k_indices,:]

    iteration = 0
    while True:
        # Iteration setup
        print('Iteration:', iteration)
        iteration += 1
        ax.clear()

        for i, k_vector in enumerate(k_vectors):
            # This labels the k_vector in the scatter plot as a red circle
            ax.scatter(k_vector[0], k_vector[1], zs=k_vector[2], s=40, c=colors[i])
        
        # Initialize clusters dictionary
        clusters = {}
        cluster_labels = {}
        for i in range(0, k):
            clusters[i] = []
            cluster_labels[i] = []

        for i, observation in enumerate(x):
            # Iterate over observations and organize cluster dict based on cluster grouping
            distances = []

            for k_vector in k_vectors:
                distances.append(np.linalg.norm(observation - k_vector))

            min_k_index = np.array(distances).argmin()
            clusters[min_k_index].append(observation)

            # Organize clusters for later purity calculation
            cluster_labels[min_k_index].append(y[i])

        cluster_purities = []
        for i, cluster in enumerate(clusters.values()):
            # Set the scatterplot
            _cluster = np.array(cluster)
            _x = _cluster[:,0]
            _y = _cluster[:,1]
            _z = _cluster[:,2]
            ax.scatter(_x, _y, zs=_z, c=colors[i], marker='x')

            # Calculate purity
            _cluster_label = np.array(cluster_labels[min_k_index])
            purity = 1 / len(_cluster_label) * max(np.unique(_cluster_label, return_counts=True)[1])
            cluster_purities.append(purity)

        # Save current cluster 3D graph to temp
        graph_filename = 'temp/iteration-{}.png'.format(str(iteration))
        plt.savefig(graph_filename)

        # Purity calculation
        purity = 0
        for i, cluster_purity in enumerate(cluster_purities):
            purity += len(cluster_labels[i]) * cluster_purity
        purity *= 1 / len(x)

        # Read graph image and write to open-cv video
        graph_image = cv2.imread(graph_filename)
        iteration_text = 'Iteration: {} Purity={}'.format(str(iteration), str(purity))
        cv2.putText(graph_image, iteration_text, (200, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2, cv2.LINE_AA)
        video.write(graph_image)

        new_k_vectors = []
        sum_l1_distance = 0
        for index, cluster in enumerate(clusters.values()):
            # Set the new k_vectors
            new_k_vector = np.mean(cluster, axis=0)
            new_k_vectors.append(new_k_vector)
            sum_l1_distance += np.linalg.norm((k_vectors[index] - new_k_vector), ord=1)

        # Check if we reach break point otherwise set new k vectors
        if sum_l1_distance <= EPSILON:
            break
        else:
            k_vectors = new_k_vectors
        
    cv2.destroyAllWindows()
    video.release()


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
    k = 2
    
    # Grab k from console
    if len(sys.argv) == 2:
        k = int(sys.argv[1])

    x, y = load_xy(csv_file)
    std_x = standardize(x)

    my_K_means(std_x, y, k)

if __name__ == '__main__':
    main()