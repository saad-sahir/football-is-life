import cv2
import numpy as np

image = cv2.imread('images/test.jpg')

def lsd(image):
    
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask_green = cv2.inRange(hsv, (36, 25, 25), (86, 255, 255))
    masked_image = cv2.bitwise_and(image, image, mask=mask_green)
    grey = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
    _, threshold = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY)
    canny = cv2.Canny(threshold, 50, 150, apertureSize=3)
    kernel = np.ones((3,3), np.uint8)
    dilated_canny = cv2.dilate(canny, kernel=kernel, iterations=1)
    closed_canny = cv2.morphologyEx(dilated_canny, cv2.MORPH_CLOSE, kernel=kernel, iterations=2)
    lines = cv2.HoughLinesP(
        closed_canny,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=50,
        maxLineGap=20,
    )
    line_image = np.zeros_like(image)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

    return lines