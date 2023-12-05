import cv2 as cv
import numpy as np
import pandas as pd
from line_model import lsd, preprocess_im
from sklearn.neighbors import NearestNeighbors


def show_image(canvas):
    cv.imshow('image', canvas)
    cv.waitKey(0)
    cv.destroyAllWindows


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


def prep_features(pitch_features, field_features):
    p = np.array(pitch_features)
    f = np.array(field_features)

    p = np.unique(p, axis=0)
    f = np.unique(f, axis=0)

    return p, f

def lines_kp(lines):
    kp = []
    for line in lines:
        kp.append(line[:2])
        kp.append(line[2:])
    return np.unique(np.array(kp), axis=0)

image = cv.imread('data/images/test/test.jpg')
pitch = cv.imread('data/images/pitch.png')

pitch_features = pd.read_csv('features/features_pitch.csv', index_col='Unnamed: 0').values
field_features = pd.read_csv('features/features_test.csv', index_col='Unnamed: 0').values

p, f = prep_features(pitch_features, field_features)

kp_p = lines_kp(p)
kp_f = lines_kp(f)