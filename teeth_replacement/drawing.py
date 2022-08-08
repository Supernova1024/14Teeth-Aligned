import cv2
import numpy as np
import mediapipe as mp

lips_connections = frozenset([
    (61, 146),
    (146, 91),
    (91, 181),
    (181, 84),
    (84, 17),
    (17, 314),
    (314, 405),
    (405, 321),
    (321, 375),
    (375, 291),
    (61, 185),
    (185, 40),
    (40, 39),
    (39, 37),
    (37, 0),
    (0, 267),
    (267, 269),
    (269, 270),
    (270, 409),
    (409, 291),
    (78, 95),
    (95, 88),
    (88, 178),
    (178, 87),
    (87, 14),
    (14, 317),
    (317, 402),
    (402, 318),
    (318, 324),
    (324, 308),
    (78, 191),
    (191, 80),
    (80, 81),
    (81, 82),
    (82, 13),
    (13, 312),
    (312, 311),
    (311, 310),
    (310, 415),
    (415, 308)])
"""MediaPipe connections for drawing lips detected by Face Mesh"""


def draw_circles(image, points, color, draw_cross=True):
    """Draw points"""
    radius = 1 if draw_cross else 0

    for point in points:
        cv2.circle(image, point, radius, color, cv2.FILLED)


def draw_line(image, line):
    """Draw a line of connected points"""
    points = np.array(line, np.int32)
    points = points.reshape((-1, 1, 2))
    cv2.polylines(image, [points], True, (0, 255, 255))


def draw_landmarks(image, landmarks, connections):
    """Draw face landmarks"""
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=landmarks,
        connections=connections,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=drawing_spec)
