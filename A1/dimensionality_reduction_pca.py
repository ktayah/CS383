# Dimensionality Reduction via PCA
# Kevin Tayah
# CS383
import os
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

def pca(data):
    standardize_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0, ddof=1)
    covariance_matrix = np.cov(standardize_data)
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

    pcaValues = []
    for index, standardize_row in enumerate(standardize_data):
        print(index, standardize_data)
        pcaValues.append(np.dot(standardize_row, eigen_vectors[index]))

    return pcaValues

def main():
    images = load_data()
    two_d_reduction = pca(images)

    plt.plot(two_d_reduction)
    plt.show()

if __name__ == "__main__":
    main()
