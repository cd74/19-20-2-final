import cv2
import math
from numba import jit
import numpy as np
import time
# def thresholdIntegral(inputMat,s,T = 0.15):
#     # outputMat=np.uint8(np.ones(inputMat.shape)*255)
#     outputMat=np.zeros(inputMat.shape)
#     nRows = inputMat.shape[0]
#     nCols = inputMat.shape[1]
#     S = int(max(nRows, nCols) / 8)

#     s2 = int(S / 4)

#     for i in range(nRows):
#         y1 = i - s2
#         y2 = i + s2

#         if (y1 < 0) :
#             y1 = 0
#         if (y2 >= nRows):
#             y2 = nRows - 1

#         for j in range(nCols):
#             x1 = j - s2
#             x2 = j + s2

#             if (x1 < 0) :
#                 x1 = 0
#             if (x2 >= nCols):
#                 x2 = nCols - 1
#             count = (x2 - x1)*(y2 - y1)

#             sum=s[y2][x2]-s[y2][x1]-s[y1][x2]+s[y1][x1]

#             if ((int)(inputMat[i][j] * count) < (int)(sum*(1.0 - T))):
#                 outputMat[i][j] = 0
#                 # print(i,j)
#             else:
#                 outputMat[i][j] = 0
#     return outputMat

# @jit(nopython=True)
def thresholdIntegral1(inputMat,s,blocksize=1,T = 0.15):
    # outputMat=np.uint8(np.ones(inputMat.shape)*255)
    outputMat=np.zeros(inputMat.shape)
    nRows = inputMat.shape[0]
    nCols = inputMat.shape[1]
    S = int(max(nRows, nCols) / 8)

    s2 = int(S*0.5*blocksize )

    for i in range(nRows):
        y1 = i - s2
        y2 = i + s2

        if (y1 < 0) :
            y1 = 0
        if (y2 >= nRows):
            y2 = nRows - 1

        for j in range(nCols):
            x1 = j - s2
            x2 = j + s2

            if (x1 < 0) :
                x1 = 0
            if (x2 >= nCols):
                x2 = nCols - 1
            count = (x2 - x1)*(y2 - y1)

            sum=s[y2][x2]-s[y2][x1]-s[y1][x2]+s[y1][x1]

            if ((int)(inputMat[i][j] * count) < (int)(sum*(1.0 - T))):
                outputMat[i][j] = 0
                # print(i,j)
            else:
                outputMat[i][j] = 255
    return outputMat

def iterateparam(list1,list2,img):
    for size in list1:
        for t in list2:
            thresh = thresholdIntegral1(img, roii,size,t)
            cv2.imwrite('results'+str(size)+'_'+str(t)+'.jpg', np.uint8(thresh))

if __name__ == '__main__':
    ratio=1
    image = cv2.imdecode(np.fromfile('test2.PNG', dtype=np.uint8), 0)
    img = cv2.resize(image, (int(image.shape[1] / ratio), int(image.shape[0] / ratio)), cv2.INTER_NEAREST)

    time_start = time.time()
    retval, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    time_end = time.time()
    print('otsu cost', time_end - time_start)
    cv2.namedWindow('OTSU threshold',0)
    cv2.imshow('OTSU threshold',otsu)
    cv2.imwrite('otsu_results.jpg',otsu)

    # thresh = cv2.adaptiveThreshold(img,  255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    # retval, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_OTSU)
    # retval, thresh = cv2.threshold(img, retval, 255, cv2.THRESH_OTSU)




    # time_start = time.time()
    # roii = cv2.integral(img)
    # time_end = time.time()
    # print('integral cost', time_end - time_start)

    # # time_start = time.time()
    # for j in range(1):
    #     thresh = thresholdIntegral1(img, roii)
    # time_end = time.time()
    # print('totally cost', time_end - time_start)
    # cv2.namedWindow('fast inergral threshold',0)
    # cv2.imshow('fast inergral threshold',thresh)
    # cv2.imwrite('results.jpg', np.uint8(thresh))


    time_start = time.time()
    roii = cv2.integral(img)
    time_end = time.time()
    print('integral cost', time_end - time_start)

    # time_start = time.time()
    for j in range(1):
        thresh = thresholdIntegral1(img, roii)
    time_end = time.time()
    print('totally cost', time_end - time_start)
    cv2.imwrite('results.jpg', np.uint8(thresh))
    cv2.namedWindow('integral threshold',0)
    cv2.imshow('integral threshold',thresh)
    list1 = [2,1,0.5]
    list2 = [0.10,0.15,0.25]
    iterateparam(list1,list2,img)


    #cv2.waitKey(0)
    cv2.destroyAllWindows()