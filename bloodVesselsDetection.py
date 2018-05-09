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

def createDataset(originalImage, expertImage, k):
    X = []
    Y = []
    black = 0
    white = 0
    flag = True
    h, w = expertImage.shape
    for i in range(0, h):
        xPos = random.randrange(k, w-k)
        yPos = random.randrange(k, h-k)
        decision = expertImage[yPos][xPos]

        if decision > 0 and abs(black - white) < 1:
            white += 1
            flag = True
        else:
            flag = False

        if decision == 0 and abs(black - white) < 1:
            black += 1
            flag = True
        else:
            flag = False

        if flag:
            r = k//2
            square = originalImage[yPos - r: yPos + r + 1, xPos - r : xPos + r + 1]
            avg = np.average(square)
            median = np.median(square)
            variance = np.var(square)
            X.append([avg, median, variance])
            Y.append(decision)

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
    for y in range(r + (3 *(originalImage.shape[0] - r)) // 5, 4 * (originalImage.shape[0] - r) // 5):
        for x in range(r + 2 * (originalImage.shape[1] - r) // 5, 4 * (originalImage.shape[1] - r) // 5):
            #print(y, x)
            square = originalImage[y - r: y + r + 1, x - r : x + r + 1]
            avg = np.average(square)
            median = np.median(square)
            variance = np.var(square)
            originalOutput[y][x] = classifier.predict([[avg, median, variance]])
        print(y, "/", 4* (originalImage.shape[0] - r) // 5)
    plt.subplot(223)
    io.imshow(originalOutput)

def main():
    fileName = "03_h.jpg"
    catalogName = "healthy"
    image = io.imread(catalogName + "/" + fileName)
    plt.subplot(221)
    io.imshow(image)
    originalImage = image[:, :, 1]
    outputImage = image[:, :, 1] #green channel
    plt.subplot(222)
    io.imshow(outputImage)
    '''image = filters.frangi(outputImage)
#    outputImage = cv2.imread(catalogName + "/" + fileName)
#    outputImage = cv2.cvtColor(outputImage, cv2.COLOR_BGR2GRAY)
#    outputImage = cv2.adaptiveThreshold(outputImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 33, 2)


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

    image2 = np.zeros_like(image)
    temp = np.zeros_like(image[0])
    i = 0
    for val in image:
        j = 0
        for x in val:
            if x > 0.0000001:
                temp[j] = 255
            else:
                temp[j] = 0
            j += 1
        image2[i] = temp
        i += 1
        temp = np.zeros_like(image[0])

    outputImage=image2
    plt.subplot(223)
    io.imshow(outputImage,cmap='gray')

'''

    expertImage = cv2.imread("healthy_manualsegm/03_h.tif")
    expertImage = cv2.cvtColor(expertImage, cv2.COLOR_BGR2GRAY)
    '''plt.subplot(224)
    io.imshow(expertImage)
    plt.show()
    compareMatrixes(expertImage, outputImage)
'''
    knn(originalImage, expertImage)



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