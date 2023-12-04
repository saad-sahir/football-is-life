import numpy as np

class CameraPoseEngine:
    def __init__(self, intrinsic_params, pan_range, tilt_range, zoom_range):
        """
        Initialize the camera pose engine.

        :param intrinsic_params: Dictionary with intrinsic parameters like focal length
        :param pan_range: Tuple (min, max) for pan angle in degrees
        :param tilt_range: Tuple (min, max) for tilt angle in degrees
        :param zoom_range: Tuple (min, max) for zoom levels
        """
        self.intrinsic_params = intrinsic_params
        self.pan_range = pan_range
        self.tilt_range = tilt_range
        self.zoom_range = zoom_range

    def generate_pose(self):
        """
        Generate a random camera pose based on the specified ranges.

        :return: Dictionary with generated camera pose parameters
        """
        pan = np.random.uniform(*self.pan_range)
        tilt = np.random.uniform(*self.tilt_range)
        zoom = np.random.uniform(*self.zoom_range)

        pose = {
            'focal_length': self.intrinsic_params['focal_length'] * zoom,
            'pan': pan,
            'tilt': tilt
        }
        return pose

# Example usage
intrinsic_params = {'focal_length': 1.0}  # Base focal length (will be scaled by zoom level)
pan_range = (-65, 65)   # Pan angle range in degrees
tilt_range = (-20, 20)  # Tilt angle range in degrees
zoom_range = (1, 5)     # Zoom level range (1x to 5x)

pose_engine = CameraPoseEngine(intrinsic_params, pan_range, tilt_range, zoom_range)
pose = pose_engine.generate_pose()
print("Generated Camera Pose:", pose)
