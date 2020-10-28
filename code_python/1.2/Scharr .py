import sys
import cv2 as cv


def main(argv):
    ## [variables]
    # First we declare the variables we are going to use
    window_name = ('Scharr - Gradient Operators to edge extraction')
    scale = 1
    delta = 0
    ddepth = cv.CV_16S
    ## [variables]

    ## [load]
    # As usual we load our source image (src)
    # Check number of arguments
    if len(argv) < 1:
        print ('Not enough parameters')
        print ('python function.py < path_to_image >')
        return -1

    # Load the image
    src = cv.imread(argv[0], cv.IMREAD_COLOR)

    # Check if image is loaded fine
    if src is None:
        print ('Error opening image: ' + argv[0])
        return -1
    

    ## [reduce_noise]

    ## [convert_to_gray]
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    ## [Scharr]
    # Gradient-X
    grad_x = cv.Scharr(gray,ddepth,1,0)
    # Gradient-Y
    grad_y = cv.Scharr(gray,ddepth,0,1)

    ## [convert]
    # converting back to uint8
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    ## [convert]

    ## [blend]
    ## Total Gradient (approximate)
    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    ## [blend]

    ## [display]
    cv.imshow(window_name, grad)
    cv.waitKey(0)
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])
