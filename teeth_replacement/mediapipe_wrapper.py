import cv2
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as point_to_pixel
from geometry_types import Point

MOUTH_RECT_INDEXES = [216, 436, 431, 211]
NOSE_WIDTH_INDEXES = [102, 331]
MOUTH_CENTER_INDEX = [13]
MOUTH_TOP_CENTER_INDEXES = [82, 312]  # [37, 267]
LEFT_EYE_INDEXES = [33, 173, 159, 145]
RIGHT_EYE_INDEXES = [398, 263, 386, 374]


def multi_face_landmarks_from_image(image_path):
    """Uses FaceMesh to detect faces on an image and returns landmarks"""
    with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:

        image = cv2.imread(image_path)
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return results.multi_face_landmarks


def convert_landmark_points_to_pixels(landmarks, landmark_indexes, image_shape):
    """Converts points from FaceMash landmark's to image's coordinate space"""
    pixels = []

    for i in landmark_indexes:
        pixel = point_to_pixel(landmarks[i].x, landmarks[i].y, image_shape[1], image_shape[0])  # what if returns none
        pixel_point = Point(pixel[0], pixel[1])
        pixels.append(pixel_point)

    return pixels
