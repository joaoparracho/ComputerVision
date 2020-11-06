import argparse
import numpy as np
import cv2
from matplotlib import pyplot as plt


def initilizeArs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default="sobel",
                        help='choose one:|sobel|scharr|prewitt|roberts|canny|laplacian|')
    parser.add_argument('--kernel_size', default=3, type=int,
                        help='Sobel and Laplacian, canny Kernel Size int')
    parser.add_argument('--input_real', type=str,
                        default="data\\harris.jpg", help='Input image path')
    return parser



def my_harris(src):
    
    blockSize = 3
    apertureSize = 3
    k = 0.04
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    ddepth = cv2.CV_64F
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, (3, 3))
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, (3, 3))
    # [convert]  converting back to uint8
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    # [blend] Total Gradient (approximate)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    # Dilate corner image to enhance corner points
    grad = cv2.dilate(grad,None)

    myHarris_dst = cv2.cornerEigenValsAndVecs(grad, blockSize, apertureSize)
    # my_Harris_dst format (λ1,λ2,x1,y1,x2,y2)

    # calculate Mc
    Mc = np.empty(grad.shape, dtype=np.float32)
    for i in range(grad.shape[0]):
        for j in range(grad.shape[1]):
            lambda_1 = myHarris_dst[i, j, 0]
            lambda_2 = myHarris_dst[i, j, 1]
            Mc[i, j] = lambda_1*lambda_2 - k*pow((lambda_1 + lambda_2), 2)

    myHarris_minVal, myHarris_maxVal, _, _ = cv2.minMaxLoc(Mc)
    #pts_Harris=np.column_stack(np.where(mask_Harris))

    myHarris_window="ola"
    myHarris_qualityLevel = 50
    max_qualityLevel = 100

    def myHarris_function(val):
        myHarris_copy = np.copy(src)
        myHarris_qualityLevel = max(val, 1)
        for i in range(grad.shape[0]):
            for j in range(grad.shape[1]):
                if Mc[i, j] > myHarris_minVal + (myHarris_maxVal - myHarris_minVal)*myHarris_qualityLevel/max_qualityLevel:
                    cv2.circle(myHarris_copy, (j, i), 4, (0,0,255) , cv2.FILLED)
        cv2.imshow(myHarris_window, myHarris_copy)

    cv2.namedWindow(myHarris_window)
    cv2.createTrackbar('Quality Level:', myHarris_window, myHarris_qualityLevel,max_qualityLevel,myHarris_function)
    myHarris_function(myHarris_qualityLevel)
    cv2.waitKey()



def run():
    opt=initilizeArs().parse_args()
    src=cv2.imread(opt.input_real)
    opt.type=opt.type.lower()

    my_harris(src)


if __name__ == '__main__':
    run()
