import cv2
import mediapipe as mp
from os import listdir
from os.path import join, isdir
from math import hypot
from collections import namedtuple
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as point_to_pixel

Point = namedtuple('Point', 'x y')


def files_list(dir_name):
    list_of_file = listdir(dir_name)
    all_files = list()

    for entry in list_of_file:
        full_path = join(dir_name, entry)

        if isdir(full_path):
            all_files = all_files + files_list(full_path)
        else:
            all_files.append(full_path)

    return all_files


def is_image(path):
    image = cv2.imread(path)
    return image is not None


def image_size(path):
    image = cv2.imread(path)

    if image is None:
        return 0
    else:
        return image.shape[0] * image.shape[1]


def landmark_from_image(image):
    with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            return None
        else:
            return results.multi_face_landmarks[0].landmark


def convert_landmark_points_to_pixels(landmarks, landmark_indexes, image_shape):
    pixels = []

    for i in landmark_indexes:
        pixel = point_to_pixel(landmarks[i].x, landmarks[i].y, image_shape[1], image_shape[0])

        if pixel is None:
            return []

        pixel_point = Point(pixel[0], pixel[1])
        pixels.append(pixel_point)

    return pixels


def is_rotated_face(landmark, image):
    pixels = convert_landmark_points_to_pixels(landmark, [34, 168, 264], image.shape)

    if not pixels:
        return True

    left_guide = pixels[0]
    mid_point = pixels[1]
    right_guide = pixels[2]

    left_dist = hypot(left_guide.x - mid_point.x, left_guide.y - mid_point.y)
    right_dist = hypot(right_guide.x - mid_point.x, right_guide.y - mid_point.y)

    if left_dist == 0 or right_dist == 0:
        return True
    elif left_dist > right_dist:
        return left_dist / right_dist > 1.5
    else:
        return right_dist / left_dist > 1.5


def face_height(landmark, image):
    points = convert_landmark_points_to_pixels(landmark, [10, 152], image.shape)

    if not points:
        return 0

    return points[1].y - points[0].y


# def mouth_height(landmark, image):
#     points = convert_landmark_points_to_pixels(landmark, [0, 17], image.shape)
#     return points[1].y - points[0].y


def is_smiling_portrait(landmark, image):
    inner_mouth_points = convert_landmark_points_to_pixels(landmark, [13, 14], image.shape)

    if not inner_mouth_points:
        return False

    inner_height = inner_mouth_points[1].y - inner_mouth_points[0].y

    face_h = face_height(landmark, image)

    if face_h <= 0:
        return False
    else:
        return inner_height / face_h > 0.0365


# def is_smiling_portrait(landmark, image):
#     face_h = face_height(landmark, image)
#     mouth_h = mouth_height(landmark, image)
#     return mouth_h / face_h > 0.12  # try 0.12 as threshold


def main():
    files = files_list('/Users/sonora/Desktop/large-dataset/imdb')
    filtered = list(filter(lambda file: is_image(file), files))  # remove files which are not images
    sorted_files = sorted(filtered, key=lambda file: image_size(file), reverse=True)  # sort by image size
    counter = 0

    for index, file in enumerate(sorted_files):
        if counter > 4000:
            break

        print(f'Processing {file}')

        image = cv2.imread(file)
        landmark = landmark_from_image(image)

        if landmark is None:
            continue

        if is_smiling_portrait(landmark, image) and not is_rotated_face(landmark, image):
            cv2.imwrite(f'/Users/sonora/Desktop/large-dataset/result/{index}.jpg', image)
            counter += 1


if __name__ == '__main__':
    main()
