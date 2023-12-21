from pitch import Pitch
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt

constants = (
    (0.16, 0.15), # penalty box constant
    (0.05, 0.23), # goalbox constant
    (0.31) # goalpost constant
)

pitch = Pitch(constants=constants)
pitch.draw(pitch.lines, pitch.points)

def apply_perspective_transformation(pitch, transformation_matrix):
    def transform_point(point, matrix):
        point = np.array([point[0], point[1], 1])
        transformed_point = np.dot(matrix, point)
        transformed_point = transformed_point / transformed_point[2]
        return (transformed_point[0], transformed_point[1])

    # Apply transformation to points
    transformed_points = {key: transform_point(point, transformation_matrix) for key, point in pitch.points.items()}

    # Apply transformation to lines
    transformed_lines = {}
    for key, line in pitch.lines.items():
        start_point, end_point = line
        transformed_start = transform_point(start_point, transformation_matrix)
        transformed_end = transform_point(end_point, transformation_matrix)
        transformed_lines[key] = (transformed_start, transformed_end)

    return {"points": transformed_points, "lines": transformed_lines}

c = 1 # random constant
z = 1 # zoom
t = 2*z*c # tilt

perspective_matrix = np.array([[z*c, t/2, 0],
                               [0, z*c, 0],
                               [0, t, c]])

# Apply the transformation
transformed_data = apply_perspective_transformation(pitch, perspective_matrix)

pitch.draw(transformed_data['lines'],transformed_data['points'])