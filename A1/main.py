# Assignment 1
# Kevin Tayah
# CS383
import os, sys
from PIL import Image

def loadData():
    # Grabs an Image instance, resizes to 40x40, and grabs the raw pixel data as a flattened 1x1600 array.
    # Returns a 2D array 156x1600 with all 152 flattened images
    images = []
    files = os.listdir('./yalefaces')

    for file in files:
        if file == 'Readme.txt':
            continue

        image = Image.open('./yalefaces/' + file).resize((40, 40))
        images.append(list(image.getdata()))
    
    return images

def main():
    images = loadData()

    print(images)

    return None

if __name__ == "__main__":
    main()
