from skimage import io
from matplotlib import pyplot as plt
import numpy as np
from math import *


def main():
    fileName = "Image_01L.jpg"
    catalogName = "CHASEDB1"
    image = io.imread(catalogName + "/" + fileName)
    plt.subplot(121)
    io.imshow(image)
    outputImage = image[:, :, 1] / 256
    plt.subplot(122)
    io.imshow(outputImage)

    plt.show()

if __name__ == '__main__':
    main()