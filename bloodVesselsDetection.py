from skimage import io, feature, measure, morphology
from matplotlib import pyplot as plt
import numpy as np
from math import *
import cv2
from skimage import filters
from sklearn.metrics import confusion_matrix
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
    specificity = tn / (fp + tn)
    accuracy = (tp + tn) / (tp + fn + fp + tn)
    print("Sensitivity = ", sensitivity, "Specificity = ", specificity, "Accuracy = ", accuracy)

def removeFrame(image, mask):
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y][x] == 0:
                image[y][x] = 0
    return image

def createDataset(originalImage, expertImage, k):
    X = []
    Y = []
    black = 0
    white = 0
    h, w = expertImage.shape
    r = k // 2
    '''
    for y in range(r, h - r):
        for x in range(r, w - r):
            decision = expertImage[y][x]
            square = originalImage[y - r: y + r + 1, x - r: x + r + 1]
            avg = np.average(square)
            median = np.median(square)
            variance = np.var(square)
            X.append([avg, median, variance])
            Y.append(decision)
        if (y % 100 == 0):
            print(y, "/", h-r)
    
    '''

    for i in range(0, h * w // 1000):
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
        square = originalImage[yPos - r: yPos + r + 1, xPos - r: xPos + r + 1]
        avg = np.average(square)
        median = np.median(square)
        variance = np.var(square)
        X.append([avg, median, variance])
        Y.append(decision)

        #if (i % 100):
        #    print(i, "/", 8000)
    
    '''while (Y.count(0) < len(Y) * 65 / 100):
        index = Y.index(255)
        Y.pop(index)
        X.pop(index)
        white -= 1
        print("white = ", white)
    '''
    #print("avg = ", avg, "median = ", median, "variance = ", variance)
    #print(square)
    #print("white = ", white, " black = ", black)
    return X, Y

def knn(originalImage, expertImage, X, Y):
    originalOutput = np.zeros_like(originalImage)
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X, Y)
    r = 5 // 2
    divideParts = 5
    #for y in range(r + (2 *(originalImage.shape[0] - r)) // divideParts, 3 * (originalImage.shape[0] - r) // divideParts):
    for y in range(r , originalImage.shape[0] - r):
    #    for x in range(r + 2 * (originalImage.shape[1] - r) // divideParts, 4 * (originalImage.shape[1] - r) // divideParts):
        for x in range(r, originalImage.shape[1] - r):
            square = originalImage[y - r: y + r + 1, x - r : x + r + 1]
            avg = np.average(square)
            median = np.median(square)
            variance = np.var(square)
            originalOutput[y][x] = classifier.predict([[avg, median, variance]])
        if (y % 500 == 0):
            print(y, "/",  (originalImage.shape[0] - r))
    return originalOutput

def main():
    exampleTab = [6, 7, 9, 10]
    learnKTT = 8

    catalogName = "healthy"
    catalogNameMask = "healthy_fovmask"
    catalogNameExpert = "healthy_manualsegm"
    learnFileName =  "0" + str(learnKTT) + "_h.jpg"
    learnFileNameExpert = "0" + str(learnKTT) + "_h.tif"

    learnImage = io.imread(catalogName + "/" + learnFileName)
    learnImage = cv2.cvtColor(learnImage, cv2.COLOR_BGR2GRAY)

    learnExpert = cv2.imread(catalogNameExpert + "/" + learnFileNameExpert)
    learnExpert = cv2.cvtColor(learnExpert, cv2.COLOR_BGR2GRAY)

    X, Y = createDataset(learnImage, learnExpert, 5)

    for example in exampleTab:
        print(example, ":")
        fileName = "0" + str(example) + "_h.jpg"
        fileNameMask = "0" + str(example) + "_h_mask.tif"
        fileNameExpert = "0" + str(example) + "_h.tif"

        resultCatalog = "outputLAST"

        image = io.imread(catalogName + "/" + fileName)
        plt.imsave(resultCatalog + "/" + str(example) + "_Example.png", image)

        #green channel
        originalImage = image[:, :, 1]
        plt.imsave(resultCatalog + "/" + str(example) + "_Green.png", originalImage, cmap='gray')

        #frangiFilter
        frangiImage = filters.frangi(originalImage)
        for y in range(frangiImage.shape[0]):
            for x in range(frangiImage.shape[1]):
                if frangiImage[y][x] > 0.0000002:
                    frangiImage[y][x] = 255
                else:
                    frangiImage[y][x] = 0
        outputImage = frangiImage

        #cut frame
        emptyMask = io.imread(catalogNameMask + "/" + fileNameMask)
        emptyMask = cv2.cvtColor(emptyMask, cv2.COLOR_BGR2GRAY)
        frangiImage = removeFrame(frangiImage, emptyMask)
        plt.imsave(resultCatalog + "/" + str(example) + "_Frangi.png", frangiImage, cmap='gray')

        expertImage = cv2.imread(catalogNameExpert + "/" + fileNameExpert)
        expertImage = cv2.cvtColor(expertImage, cv2.COLOR_BGR2GRAY)
        plt.imsave(resultCatalog + "/" + str(example) + "_ExpertImage.png", expertImage, cmap='gray')

        print("Frangi:")
        compareMatrixes(expertImage, outputImage)

        r = 5 // 2
        divideParts = 5
        #squareExpertImage = expertImage[r + (3 * (originalImage.shape[0] - r)) // divideParts : 4 * (originalImage.shape[0] - r) // divideParts,
                            #r + 2 * (originalImage.shape[1] - r) // divideParts :  4 * (originalImage.shape[1] - r) // divideParts]
        knnImage = knn(originalImage, expertImage, X, Y)
        plt.imsave(resultCatalog + "/" + str(example) + "_knnImage.png", knnImage, cmap='gray')

        #knnImage = knnImage[r + (3 * (ori ginalImage.shape[0] - r)) // divideParts : 4 * (originalImage.shape[0] - r) // divideParts,
                            #r + 2 * (originalImage.shape[1] - r) // divideParts :  4 * (originalImage.shape[1] - r) // divideParts]
        #print(squareExpertImage.shape, knnImage.shape)

        knnImageAfterErosion = morphology.erosion(knnImage)
        plt.imsave(resultCatalog + "/" + str(example) + "_knnImageAfterErosion.png", knnImageAfterErosion, cmap='gray')
        print("kNN:")
        compareMatrixes(expertImage, knnImageAfterErosion)

if __name__ == '__main__':
    main()