from skimage import io, feature, measure
from matplotlib import pyplot as plt
import numpy as np
from math import *
from scipy import ndimage
import cv2
from skimage.draw import circle
from skimage.filters import frangi

def cutCircle(image):
    img = np.zeros_like(image)
    h, w = image.shape
    r = min(h, w) / 2
    center = h / 2 - 1, w / 2 - 1
    rr, cc = circle(center[0], center[1], r)
    img[rr, cc] = 1
    return img


def main():
    fileName = "myPhoto.jpg"
    catalogName = "CHASEDB1"
    image = io.imread(catalogName + "/" + fileName)
    plt.subplot(221)
    io.imshow(image)

    outputImage = image[:, :, 1] #green channel
    plt.subplot(222)
    io.imshow(outputImage)

    outputImage = cv2.imread(catalogName + "/" + fileName)
    outputImage = cv2.cvtColor(outputImage, cv2.COLOR_BGR2GRAY)
    outputImage = cv2.Canny(outputImage, 150, 255, apertureSize=5)
    myMask = np.zeros((3, 3), np.uint8)
    myMask[1,:] = 1
    myMask[:,1] = 1
    outputImage = cv2.dilate(outputImage, myMask, iterations=3)
    outputImage = cv2.medianBlur(outputImage, 3)
    outputImage = cv2.erode(outputImage, myMask, iterations=3)

    #outputImage = cv2.adaptiveThreshold(outputImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    #outputImage = cv2.erode(outputImage, np.ones((3,3), np.uint8), iterations=3)
    #outputImage = cv2.medianBlur(outputImage, 5)
    #outputImage = cv2.Canny(outputImage,)
    plt.subplot(223)
    io.imshow(outputImage)
    outputImage = frangi(outputImage, black_ridges=True)
    plt.subplot(224)

    '''
    for y in range(outputImage.shape[0]):
        for x in range(outputImage.shape[1]):
            if (outputImage[y][x] != 0):
                outputImage[y][x] = 1
    '''
    #io.imshow(outputImage, cmap=plt.get_cmap('gray'))
    io.imshow(outputImage)
    #outputImage = feature.canny(outputImage, sigma=0.5)

    #contours = measure.find_contours(outputImage, 0.3)
    
    plt.show()

if __name__ == '__main__':
    main()