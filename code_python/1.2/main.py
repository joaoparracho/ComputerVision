import argparse
import numpy as np
import cv2


def initilizeArs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default="canny",
                        help='sobel|scharr|prewitt|roberts|canny|laplacian|')
    parser.add_argument('--input_real', type=str,
                        default="data\\img15.jpg", help='Input image')
    return parser


def sobel(src):
    window_name = ('Sobel Demo - Gradient Operators to edge extraction')
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    # [convert_to_gray]
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # [sobel]
    # Gradient-X
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale,delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale,delta=delta, borderType=cv2.BORDER_DEFAULT)
    # [convert]
    # # converting back to uint8
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    # [blend]
    # Total Gradient (approximate)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    # [display]
    cv2.imshow(window_name, grad)

def scharr(src):
    window_name = ('Scharr - Gradient Operators to edge extraction')
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    ## [convert_to_gray]
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ## [Scharr]
    # Gradient-X
    grad_x = cv2.Scharr(gray,ddepth,1,0)
    # Gradient-Y
    grad_y = cv2.Scharr(gray,ddepth,0,1)
    ## [convert]
    # converting back to uint8
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    ## [blend]
    ## Total Gradient (approximate)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    ## [display]
    cv2.imshow(window_name, grad)

def prewitt(src):
    window_name = ('prewiit - Gradient Operators to edge extraction')
    # [convert_to_gray]
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    img_prewittx = cv2.filter2D(gray, -1, kernelx)
    img_prewitty = cv2.filter2D(gray, -1, kernely)

    # [display]
    cv2.imshow("Prewitt X", img_prewittx)
    cv2.imshow("Prewitt Y", img_prewitty)
    cv2.imshow(window_name, img_prewittx + img_prewitty)
    
def roberts(src):
    window_name = ('roberts - Gradient Operators to edge extraction')
    # [convert_to_gray]
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    kernelx = np.array([[1,0],[0,-1]])
    kernely = np.array([[0,1],[-1,0]])
    img_Robertsx = cv2.filter2D(gray, -1, kernelx)
    img_Robertsy = cv2.filter2D(gray, -1, kernely)
    # [display]
    cv2.imshow("Roberts X", img_Robertsx)
    cv2.imshow("Roberts Y", img_Robertsy)
    cv2.imshow(window_name, img_Robertsx + img_Robertsy)

def canny(src):
    def runn(val):
        low_threshold = val 
        detected_edges = cv2.Canny(gray, low_threshold, low_threshold*ratio, kernel_size)
        mask = detected_edges != 0
        dst = src * (mask[:,:,None].astype(src.dtype))
        cv2.imshow(window_name, dst)
        
    window_name = ('canny - Gradient Operators to edge extraction')
    max_lowThreshold = 100
    title_trackbar = 'Min Threshold:'
    ratio = 3
    kernel_size = 3
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow(window_name)
    cv2.createTrackbar(title_trackbar, window_name , 0, max_lowThreshold, runn)
    runn(0)

def Laplacian(src):
    # [variables]
    # Declare the variables we are going to use
    window_name = ('Laplace - Gradient Operators to edge extraction')
    ddepth = cv2.CV_16S
    kernel_size = 3
    # [convert_to_gray]
    # Convert the image to grayscale
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # [laplacian]
    # Apply Laplace function
    dst = cv2.Laplacian(src_gray, ddepth, ksize=kernel_size)
    # [convert]
    # converting back to uint8
    abs_dst = cv2.convertScaleAbs(dst)
    # [display]
    cv2.imshow(window_name, abs_dst)

def run():
    opt = initilizeArs().parse_args()
    src = cv2.imread(opt.input_real)

    if opt.type == "sobel": sobel(src)
    elif (opt.type == "scharr"): scharr(src)
    elif opt.type == "prewitt": prewitt(src)
    elif opt.type == "roberts": roberts(src)
    elif opt.type == "canny": canny(src)
    elif opt.type == "laplacian": Laplacian(src)

    

    cv2.waitKey(0)~
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()
