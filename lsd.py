import numpy as np
import cv2 

def preprocess_image(image_path, show=False):
    """
    Params:
    image_path: path of the image lmao
    """
    image = cv2.imread(image_path)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask_green = cv2.inRange(hsv, (36, 25, 25), (86, 255, 255))
    masked_image = cv2.bitwise_and(image, image, mask=mask_green)
    grey = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
    _, threshold = cv2.threshold(grey, 135, 255, cv2.THRESH_BINARY)
    canny = cv2.Canny(threshold, 40, 180, apertureSize=3)
    kernel = np.ones((5,5), np.uint8)
    dilated_canny = cv2.dilate(canny, kernel=kernel, iterations=1)

    lines = cv2.HoughLinesP(
        dilated_canny,
        rho=1,
        theta=np.pi/180,
        threshold=350,
        minLineLength=100,
        maxLineGap=100,
    )

    line_image = np.zeros_like(image)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 1)

    if show:
        target = masked_image
        cv2.imshow('Image', target)
        cv2.imwrite('results/mask_test3.png', target)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return line_image, lines

# Demo
image_path = 'data/images/test/test3.jpg'
preprocess_image(image_path, show=True)