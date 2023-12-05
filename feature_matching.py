from line_model import lsd, preprocess_im
import cv2 as cv
import numpy as np
import pandas as pd

def show_image(canvas):
    cv.imshow('image', canvas)
    cv.waitKey(0)
    cv.destroyAllWindows

def draw_lines(lines, canvas):
    for line in lines:
        y1, x1, y2, x2 = line
        cv.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 1)

    show_image(canvas)

def radar_features(radar, save=False):
    pitch = lsd(radar)
    if save:
        df = pd.DataFrame(pitch, columns=['y1','x1','y2','x2'])
        df.to_csv('features_pitch.csv')
    return pitch

def image_features(im, save=False):
    image_features = lsd(im)
    if save:
        df = pd.DataFrame(image_features, columns=['y1','x1','y2','x2'])
        df.to_csv('features_test.csv')
    return image_features

image = cv.imread('images/test/test.jpg')
pitch = cv.imread('images/pitch.png')

pitch_features = np.array(pd.read_csv('features/features_pitch.csv').values[:, 1:])
field_features = np.array(pd.read_csv('features/features_test.csv').values[:, 1:])

h_pitch, w_pitch = pitch.shape[:2]
h_image, w_image = image.shape[:2]

normalize_points = lambda points, height, width: [
    (y1 / height, x1 / width, y2 / height, x2 / width)
    for y1, x1, y2, x2 in points
]

norm_pitch = normalize_points(pitch_features, h_pitch, w_pitch)
norm_image = normalize_points(image_features, h_image, w_image)
points_pitch = [(x1, y1) for _, x1, _, y1 in norm_pitch]
points_test = [(x2, y2) for _, x2, _, y2 in norm_image]
points_pitch = np.array(points_pitch, dtype=np.float32)
points_test = np.array(points_test, dtype=np.float32)

H, _ = cv.findHomography(points_pitch, points_test, cv.RANSAC)

h, w = image.shape[:2]
warped = cv.warpPerspective(pitch, H, (w, h))

show_image(warped)