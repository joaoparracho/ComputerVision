import cv2
import numpy as np

img = cv2.imread('data\\img15.jpg')
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img_gaussian = cv2.GaussianBlur(gray,(3,3),0)


#prewitt
kernelx = np.array([[1,0],[0,-1]])
kernely = np.array([[0,1],[-1,0]])
print(kernelx)
img_Robertsx = cv2.filter2D(img, -1, kernelx)
img_Robertsy = cv2.filter2D(img, -1, kernely)

cv2.imshow("Roberts X", img_Robertsx)
cv2.imshow("Roberts Y", img_Robertsy)
cv2.imshow("Roberts", img_Robertsx + img_Robertsy)
cv2.waitKey()
cv2.destroyAllWindows()