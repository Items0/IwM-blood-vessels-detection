from skimage import io, feature, measure, morphology
from matplotlib import pyplot as plt
import numpy as np
from math import *
import cv2
from skimage.draw import circle
from skimage import filters
from sklearn.metrics import confusion_matrix
import pandas as pd
import random
from sklearn.neighbors import KNeighborsClassifier
import sys

def removeFrame(image, mask):
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y][x] == 0:
                image[y][x] = 0
    return image

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
    specificity = tn / (fp + tn)
    accuracy = (tp + tn) / (tp + fn + fp + tn)
    print("Sensitivity = ", sensitivity, "Specificity = ", specificity, "Accuracy = ", accuracy)

def createDataset(originalImage, expertImage, k):
    X = []
    Y = []
    black = 0
    white = 0
    h, w = expertImage.shape
    for i in range(0, 8000):
        while True:
            xPos = random.randrange(k, w - k)
            yPos = random.randrange(k, h - k)
            decision = expertImage[yPos][xPos]
            if decision == 255 * (i % 2):
                break

        if decision > 0:
            white += 1
        else:
            black += 1

        r = k // 2
        square = originalImage[yPos - r: yPos + r + 1, xPos - r : xPos + r + 1]
        avg = np.average(square)
        median = np.median(square)
        variance = np.var(square)
        X.append([avg, median, variance])
        Y.append(decision)
        if (i % 100):
            print(i, "/", 8000)
    while (Y.count(0) < len(Y) * 65 / 100):
        index = Y.index(255)
        Y.pop(index)
        X.pop(index)
        white -= 1
        print("white = ", white)

    #print("avg = ", avg, "median = ", median, "variance = ", variance)
    #print(square)
    print("white = ", white, " black = ", black)
    return X, Y

def knn(originalImage, expertImage):
    originalOutput = np.zeros_like(originalImage)
    names = ['avg', 'median', 'variance']
    X, Y = createDataset(originalImage, expertImage, 5)
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X, Y)
    r = 5 // 2
    divideParts = 5
    for y in range(r + (3 *(originalImage.shape[0] - r)) // divideParts, 4 * (originalImage.shape[0] - r) // divideParts):
        for x in range(r + 2 * (originalImage.shape[1] - r) // divideParts, 4 * (originalImage.shape[1] - r) // divideParts):
            square = originalImage[y - r: y + r + 1, x - r : x + r + 1]
            avg = np.average(square)
            median = np.median(square)
            variance = np.var(square)
            originalOutput[y][x] = classifier.predict([[avg, median, variance]])
        print(y, "/", 4* (originalImage.shape[0] - r) // 5)
    plt.subplot(223)
    io.imshow(originalOutput)

    originalOutput = morphology.erosion(originalOutput)
    plt.subplot(224)
    io.imshow(originalOutput)

    return originalOutput

def main():
    fileName = "08_h.jpg"
    catalogName = "healthy"
    image = io.imread(catalogName + "/" + fileName)
    plt.subplot(221)
    io.imshow(image)
    originalImage = image[:, :, 1]
    outputImage = image[:, :, 1] #green channel
    plt.subplot(222)
    io.imshow(outputImage)
    image = filters.frangi(outputImage)

    image2 = np.zeros_like(image)
    temp = np.zeros_like(image[0])
    i = 0
    for val in image:
        j = 0
        for x in val:
            if x > 0.0000002:
                temp[j] = 255
            else:
                temp[j] = 0
            j += 1
        image2[i] = temp
        i += 1
        temp = np.zeros_like(image[0])

    outputImage = image2
    fileNameMask = "06_h_mask.tif"
    emptyMask = io.imread("healthy_fovmask" + "/" + fileNameMask)
    emptyMask = cv2.cvtColor(emptyMask, cv2.COLOR_BGR2GRAY)
    outputImage = removeFrame(outputImage, emptyMask)
    #plt.subplot(223)
    #io.imshow(outputImage, cmap='gray')

    expertImage = cv2.imread("healthy_manualsegm/08_h.tif")
    expertImage = cv2.cvtColor(expertImage, cv2.COLOR_BGR2GRAY)
    #plt.subplot(222)
    #io.imshow(expertImage)
    #plt.show()
    compareMatrixes(expertImage, outputImage)

    r = 5 // 2
    divideParts = 5
    squareExpertImage = expertImage[r + (3 * (originalImage.shape[0] - r)) // divideParts : 4 * (originalImage.shape[0] - r) // divideParts,
                        r + 2 * (originalImage.shape[1] - r) // divideParts :  4 * (originalImage.shape[1] - r) // divideParts]
    knnImage = knn(originalImage, expertImage)
    knnImage = knnImage[r + (3 * (originalImage.shape[0] - r)) // divideParts : 4 * (originalImage.shape[0] - r) // divideParts,
                        r + 2 * (originalImage.shape[1] - r) // divideParts :  4 * (originalImage.shape[1] - r) // divideParts]
    print(squareExpertImage.shape, knnImage.shape)
    compareMatrixes(squareExpertImage, knnImage)

    plt.show()

if __name__ == '__main__':
    main()