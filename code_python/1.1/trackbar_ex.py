import cv2


def bar(x):
    """ Print the value of the trackbar whenever it changes """
    print(f"value: {x}")


img = cv2.imread("data\\img15_noise.jpg", cv2.IMREAD_GRAYSCALE)

thresholdSteps = 10
beginningTrackbarValue = 3  # Used in the call to createTrackbar to set the initial trackbar value
cv2.namedWindow("Image")
cv2.createTrackbar("trackbar1", "Image", beginningTrackbarValue, thresholdSteps, bar)

previousTrackbarValue = -1  # Set this to -1 so the threshold will be applied and the image displayed the first time through the loop
while True:
    newTrackBarValue = cv2.getTrackbarPos("trackbar1", "Image")
    # Don't process the image if the trackbar value is still the same
    if newTrackBarValue != previousTrackbarValue:
        thresholdValue = newTrackBarValue * 255 / thresholdSteps
        print(f"threshold value: {thresholdValue}")
        _, threshImg = cv2.threshold(img, thresholdValue, 255, cv2.THRESH_BINARY)
        cv2.imshow("Image", threshImg)
        previousTrackbarValue = newTrackBarValue
    key = cv2.waitKey(10)
    if key == 27:
        break