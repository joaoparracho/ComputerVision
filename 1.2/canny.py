import cv2
import numpy as np

img = cv2.imread('data\\img15.jpg')


img_canny = cv2.Canny(img,100,200)

cv2.imshow("Canny", img_canny)
cv2.waitKey()
cv2.destroyAllWindows()