# Dimensionality Reduction via PCA
# Kevin Tayah
# CS383
import os, sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_data():
    # Grabs an Image instance, resizes to 40x40, and grabs the raw pixel data as a flattened 1x1600 array.
    # Returns a 2D array 156x1600 with all 152 flattened images
    images = []
    files = os.listdir('./yalefaces')

    for file in files:
        if file == 'Readme.txt':
            continue

        image = list(Image.open('./yalefaces/' + file).resize((40, 40)).getdata())
        images.append(image)
    return images

def unstandardize(data, orig):
    return np.around(data * np.std(orig, axis=0, ddof=1) + np.mean(orig, axis=0)).real

def standardize(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0, ddof=1)
    

def pca(data, n=2):
    standardize_data = standardize(data)
    covariance_matrix = np.cov(standardize_data.T, ddof=1)
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

    max_eigen_value_indices = (-eigen_values).argsort()[:n] # Finds indices of n highest eigen_values

    pca = []
    for max_eigen_value_index in max_eigen_value_indices:
        pca.append(np.dot(standardize_data, eigen_vectors[:,max_eigen_value_index]))

    return np.array(pca)

def dimensionality_reduction_pca():
    images = load_data()
    two_d_reduction_x, two_d_reduction_y = pca(images)

    plt.plot(two_d_reduction_x, two_d_reduction_y, 'ro')
    plt.show()

def dimensionality_reduction_reconstruction():
    images = load_data()

    video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'mp4v'), 1, (40, 40))

    standardize_data = standardize(images)
    covariance_matrix = np.cov(standardize_data.T, ddof=1)
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

    for k in range(1, 1601):
        print('Reconstructing with ' + str(k) + ' of the most relevant eigenvectors.')
        max_eigen_value_indices = (-eigen_values).argsort()[:k] # Finds indices of n highest eigen_values

        pca = []
        relevantVectors = []

        for max_eigen_value_index in max_eigen_value_indices:
            relevantVector = eigen_vectors[:,max_eigen_value_index]
            relevantVectors.append(relevantVector)
            pca.append(np.dot(standardize_data, relevantVector))

        np_relevantVectors = np.array(relevantVectors).T # Transpose it so our vectors fall along the columns
        np_pca = np.array(pca)

        reconstruction_standardized = np.dot(np_pca.T, np_relevantVectors.T) # Based on equation, x^ = zW^T
        reconstruction = unstandardize(reconstruction_standardized, images)
        reshaped = reconstruction[0].reshape((40, 40)).astype(np.uint8)

        img_filename = './temp/' + str(k) + '.jpeg'
        Image.fromarray(reshaped).save(img_filename, 'JPEG', quality = 95)

        video.write(cv2.imread(img_filename))

    cv2.destroyAllWindows()
    video.release()
        
if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'p2':
        dimensionality_reduction_pca()
    elif len(sys.argv) == 2 and sys.argv[1] == 'p3':
        dimensionality_reduction_reconstruction()
    else:
        print('Incorrect usage. Execute "python main.py p2" for problem 2 and "python main.py p3" for problem 3')