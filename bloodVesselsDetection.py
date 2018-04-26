from skimage import io, feature, measure, morphology
from matplotlib import pyplot as plt
import numpy as np
from math import *
import cv2
from skimage.draw import circle
from skimage.filters import frangi
from sklearn.metrics import confusion_matrix

def compareMatrixes(expert, algorithm):
    '''
            \ Algorithm
     Expert  \   1  0
              \_________
            1 |  a||b       TP||FN
              |--------     ------
            0 |  c||d       FP||TN
    '''
    expert1d = expert.ravel()
    algorithm1d = algorithm.ravel()
    tn, fp, fn, tp = confusion_matrix(expert1d, algorithm1d).ravel()
    #print(tn, fp, fn, tp)
    sensitivity = tp / (tp + fn)
    specificity = fn / (fp + tn)
    accuracy = (tp + tn) / (tp + fn + fp + tn)
    print("Sensitivity = ", sensitivity, "Specificity = ", specificity, "Accuracy = ", accuracy)

def main():
    fileName = "08_h.jpg"
    catalogName = "healthy"
    image = io.imread(catalogName + "/" + fileName)
    plt.subplot(221)
    io.imshow(image)

    outputImage = image[:, :, 1] #green channel
    plt.subplot(222)
    io.imshow(outputImage)

    outputImage = cv2.imread(catalogName + "/" + fileName)
    outputImage = cv2.cvtColor(outputImage, cv2.COLOR_BGR2GRAY)
    outputImage = cv2.adaptiveThreshold(outputImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 33, 2)

    myMask = np.zeros((3, 3), np.uint8)
    myMask[1, :] = 1
    myMask[:, 1] = 1

    #outputImage = cv2.medianBlur(outputImage, 9)
    outputImage = morphology.dilation(outputImage)
    outputImage = morphology.dilation(outputImage)
    outputImage = morphology.erosion(outputImage)
    outputImage = morphology.erosion(outputImage)
    outputImage = cv2.medianBlur(outputImage, 11)
    outputImage = cv2.medianBlur(outputImage, 11)
    outputImage = cv2.medianBlur(outputImage, 11)
    #outputImage = cv2.medianBlur(outputImage, 5)
    outputImage = cv2.bitwise_not(outputImage)
    plt.subplot(223)
    io.imshow(outputImage)


    expertImage = cv2.imread("healthy_manualsegm/08_h.tif")
    expertImage = cv2.cvtColor(expertImage, cv2.COLOR_BGR2GRAY)
    plt.subplot(224)
    io.imshow(expertImage)

    compareMatrixes(expertImage, outputImage)

    '''
    outputImage = cv2.medianBlur(outputImage, 11)
    outputImage = frangi(outputImage, 30, 200,black_ridges=True)
    outputImage = cv2.Canny(outputImage, threshold1=150, threshold2=255, apertureSize=7, L2gradient=True)

    
    outputImage = cv2.dilate(outputImage, myMask, iterations=3)
    outputImage = cv2.erode(outputImage, myMask, iterations=3)
    outputImage = cv2.adaptiveThreshold(outputImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    '''
    plt.show()

if __name__ == '__main__':
    main()