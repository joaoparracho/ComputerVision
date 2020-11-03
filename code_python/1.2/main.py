import argparse
import numpy as np
import cv2
from matplotlib import pyplot as plt


def initilizeArs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default="scharr",
                        help='sobel|scharr|prewitt|roberts|canny|laplacian|')
    parser.add_argument('--kernel_size', default=3,type=int, help='Sobel anda Laplacian, canny Kernel Size')
    parser.add_argument('--lower', default=100,type=int, help='Canny low threshold')
    parser.add_argument('--upper', default=200,type=int, help='Canny max threshold')
    parser.add_argument('--input_real', type=str,
                        default="data\\img15.jpg", help='Input image')
    return parser

def sobel(src,kernelSize):
    window_name = f"Gradient Operators to edge extraction\nSobel KernelSize={kernelSize}"
    ddepth = -1
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray,ddepth,1,0,ksize=kernelSize)
    grad_y = cv2.Sobel(gray,ddepth,0,1,ksize=kernelSize)
    # [convert]  converting back to uint8
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    # [blend] Total Gradient (approximate)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return[abs_grad_x,abs_grad_y,grad,window_name]

def scharr(src):
    window_name = f"Gradient Operators to edge extraction\nScharr"
    ddepth = -1
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Scharr(gray,ddepth,1,0)
    grad_y = cv2.Scharr(gray,ddepth,0,1)
    ## [convert] converting back to uint8
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    ## [blend] Total Gradient (approximate)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return[abs_grad_x,abs_grad_y,grad,window_name]

def prewitt(src):
    window_name = f"Gradiprewittent Operators to edge extraction\nPrewitt"
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    kernelx = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    #ddepth=-1, the output image will have the same depth as the source.
    img_prewittx = cv2.filter2D(gray,-1, kernelx)
    img_prewitty = cv2.filter2D(gray,-1, kernely)
    return[img_prewittx,img_prewitty,img_prewittx + img_prewitty,window_name]
    
def roberts(src):
    window_name = f"Gradiprewittent Operators to edge extraction\nRoberts"
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    kernelx = np.array([[-1,0],[0,1]])
    kernely = np.array([[0,-1],[1,0]])
    #ddepth=-1, the output image will have the same depth as the source.
    img_Robertsx = cv2.filter2D(gray, -1, kernelx)
    img_Robertsy = cv2.filter2D(gray, -1, kernely)

    return[img_Robertsx,img_Robertsy,img_Robertsx + img_Robertsy,window_name]

def canny(src,lower,upper,kernelSize):
    window_name = f"Gradient Operators to edge extraction\nCanny aperturesize={kernelSize}\n lower={lower} | upper={upper} "
    img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img,lower,upper,apertureSize=kernelSize)
    return[edges,window_name]

def Laplacian(src,kernelSize):
    window_name = f"Gradient Operators to edge extraction\nLaplace KernelSize={kernelSize} "
    ddepth = -1
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    dst = cv2.Laplacian(src_gray, ddepth, ksize=kernelSize)
    # [convert]# converting back to uint8
    abs_dst = cv2.convertScaleAbs(dst)
    return[abs_dst,window_name]

def run():
    opt = initilizeArs().parse_args()
    src = cv2.imread(opt.input_real)
    opt.type=opt.type.lower()
    try:
        if opt.type == "sobel": 
            try:
                   [Kernelx,Kernely,Kernelxy,window_name]=sobel(src,opt.kernel_size)
            except:
             print("Sobel Kernel Size arg int >0 \nArgs problem, use --help arg")
        elif (opt.type == "scharr"):[Kernelx,Kernely,Kernelxy,window_name]=scharr(src)
        elif opt.type == "prewitt": [Kernelx,Kernely,Kernelxy,window_name]=prewitt(src)
        elif opt.type == "roberts": [Kernelx,Kernely,Kernelxy,window_name]=roberts(src)
        elif opt.type == "canny":  
            try:
                   [Kernelxy,window_name]=canny(src,opt.lower,opt.upper,opt.kernel_size)
            except:
             print("Canny need args low and mad threshold\nArgs problem, use --help arg")    
        elif opt.type == "laplacian":
            try:
                   [Kernelxy,window_name]=Laplacian(src,opt.kernel_size)
            except:
             print("LaPlacian Kernel Size arg int\nArgs problem, use --help arg")
    except:
         print("Args problem, use --help arg")
    try:
        if(opt.type == "sobel" or opt.type == "scharr" or opt.type == "prewitt" or opt.type == "roberts"):
            plt.suptitle(window_name)
            plt.subplot(221), plt.imshow(src,'gray'), plt.title("Original Image")
            plt.xticks([]), plt.yticks([])
            plt.subplot(222), plt.imshow(Kernelxy,'gray'), plt.title(opt.type.upper())
            plt.xticks([]), plt.yticks([])
            plt.subplot(223), plt.imshow(Kernelx,'gray'), plt.title(f"{opt.type.upper()} X")
            plt.xticks([]), plt.yticks([])
            plt.subplot(224), plt.imshow(Kernely,'gray'), plt.title(f"{opt.type.upper()} Y")
            plt.xticks([]), plt.yticks([])
            plt.show()
        elif (opt.type == "laplacian" or opt.type == "canny" ):
            plt.suptitle(window_name)
            plt.subplot(121), plt.imshow(src,'gray'), plt.title("Original Image")
            plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(Kernelxy,'gray'), plt.title(opt.type.upper())
            plt.xticks([]), plt.yticks([])
            plt.show()
            
    except:
        prinr("IMG-Ploting error")


if __name__ == '__main__':
    run()
