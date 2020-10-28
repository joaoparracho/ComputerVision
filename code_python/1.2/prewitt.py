import sys
import cv2 as cv
import numpy as np

def main(argv):
    # [variables]
    # First we declare the variables we are going to use
    window_name = ('Scharr - Gradient Operators to edge extraction')
    # [load]
    # As usual we load our source image (src)
    # Check number of arguments
    if len(argv) < 1:
        print('Not enough parameters')
        print('python function.py < path_to_image >')
        return -1

    # Load the image
   # src = cv.imread(argv[0], cv.IMREAD_COLOR)
    src = cv.imread('data\\img15.jpg', cv.IMREAD_COLOR)

    # Check if image is loaded fine
    if src is None:
        print('Error opening image: ' + argv[0])
        return -1

    # [reduce_noise]

    # [convert_to_gray]
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    img_prewittx = cv.filter2D(gray, -1, kernelx)
    img_prewitty = cv.filter2D(gray, -1, kernely)

    # [convert]
    # converting back to uint8
    abs_grad_x = cv.convertScaleAbs(img_prewittx)
    abs_grad_y = cv.convertScaleAbs(img_prewitty)

    # [blend]
    # Total Gradient (approximate)
    grad = cv.addWeighted(abs_grad_x, 1, abs_grad_y, 1, 0)

    # [display]
    cv.imshow("Prewitt X", img_prewittx)
    cv.imshow("Prewitt Y", img_prewitty)
    cv.imshow("Prewitt", img_prewittx + img_prewitty)
    cv.imshow(window_name, grad)
    cv.waitKey(0)
    return 0

if __name__ == "__main__":
    main("c:\\Users\\Anastacio\\Desktop\\VC\\pratica\\proj1\\data\\img15.jpg")
