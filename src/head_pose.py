import cv2
import numpy as np


class HeadPoseEstimator:

    def estimate_pose(self, landmarks, frame_shape):

        h, w = frame_shape[:2]

        image_points = np.array([
            landmarks[1],      # Nose tip
            landmarks[152],    # Chin
            landmarks[33],     # Left eye corner
            landmarks[263],    # Right eye corner
            landmarks[61],     # Left mouth corner
            landmarks[291]     # Right mouth corner
        ], dtype=np.float64)

        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1)
        ])

        focal_length = w

        camera_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs
        )

        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

        pitch = angles[0]
        yaw = angles[1]
        roll = angles[2]

        # Normalize pitch

        if pitch > 90:
            pitch = pitch - 180

        elif pitch < -90:
            pitch = pitch + 180

        return pitch, yaw, roll