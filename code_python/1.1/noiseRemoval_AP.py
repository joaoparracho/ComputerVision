import argparse
import numpy as np
import cv2
import math
max_type = 3
max_value = 50
trackbar_type = "Mean\nGaussian\nMedian\nBilateral"
trackbar_value = "Kernel"
window_name = 'Noise Removal"'
input_noise=0

class runGUI():
    def __init__(self,org,noise):
        self.noise = noise
        self.org = org
    def run(self,a):
        filter_type = cv2.getTrackbarPos(trackbar_type, window_name)
        kernel_value = cv2.getTrackbarPos(trackbar_value, window_name)
        kernel_value = kernel_value - (1-kernel_value%2) if kernel_value!=0 else 1
        result=denoise_filters(self.noise,str(filter_type),kernel_value)
        cv2.imshow("Input image", self.org)
        cv2.imshow("Noise image", self.noise)
        cv2.imshow("Filtered Image", result)

def initilizeArs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter',type=str,default="mean", help='mean|median|gaussian|bilateral')
    parser.add_argument('--input_noise',type=str,default="data\\img15_noise.jpg", help='Input image')
    parser.add_argument('--input_real',type=str,default="data\\img15.jpg", help='Input image')
    parser.add_argument('--kernel_size',type=int,default=3, help='Kernel Size')
    parser.add_argument('--gui', default=True,action='store_true', help='Use Graphic User Interface')
    return parser

def denoise_filters(src,filter_s,kernel):
    if(filter_s=="mean" or filter_s=="0"): return cv2.blur(src, (kernel, kernel))
    elif(filter_s=="gaussian" or filter_s=="1"): return cv2.GaussianBlur(src, (kernel, kernel), 0)
    elif(filter_s=="median" or filter_s=="2"): return cv2.medianBlur(src, kernel)
    elif(filter_s=="bilateral" or filter_s=="3"): return cv2.bilateralFilter(src, kernel, kernel * 2, kernel* 2)
    return None

def runFilter(src,src_r,filter_type,kernel_size):
    result=denoise_filters(src,filter_type,kernel_size)
    cv2.imshow("Noise image",src)
    cv2.imshow("Filtered image with "+filter_type+" kernerl_size=="+str(kernel_size),result)
    cv2.imshow("Original image",src_r)
def run():
    opt=initilizeArs().parse_args()
    src = cv2.imread(opt.input_noise)
    src_r = cv2.imread(opt.input_real)
    if(opt.gui):
        gui=runGUI(src_r,src)
        cv2.namedWindow(window_name,cv2.WINDOW_FULLSCREEN)
        cv2.createTrackbar(trackbar_type, window_name , 0, max_type, gui.run)
        cv2.createTrackbar(trackbar_value, window_name , 1, max_value, gui.run)
        gui.run(0)
    else: runFilter(src,src_r,opt.filter,opt.kernel_size)

    cv2.waitKey(0)
    
if __name__ == '__main__':
    run()

