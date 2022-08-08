import os
import numpy
from math import hypot, atan2, pi
from PIL import Image, ImageFilter
from script_location import script_location as location
from lips_detector import inner_mouth_mask_from_image
from MouthAnalyzer import MouthAnalyzer, GumType, GumData
from ImageProvider import TeethImageProvider, TeethImageType
from geometry_types import *
from mediapipe_wrapper import *
from image_operations import *

_OUTPUT_IMAGE_PATH = f'{location}/images/output/output.png'


def _flattened_contours(contours) -> list[list[Point]]:
    """Flatten OpenCV contours into arrays of points"""
    flattened = []

    for contour in contours:
        flattened_contour = []

        for contour_point in contour[:, 0]:
            point = Point(contour_point[0], contour_point[1])
            flattened_contour.append(point)

        flattened.append(flattened_contour)

    return flattened


def _contour_from_mask(mask) -> list[Point]:
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    flattened = _flattened_contours(contours)
    return max(flattened, key=lambda c: len(c))


def _cropped_mouth_image(landmark, image):
    rect_pixels = convert_landmark_points_to_pixels(landmark, MOUTH_RECT_INDEXES, image.shape)

    top_left = rect_pixels[0]
    top_right = rect_pixels[1]
    bottom_right = rect_pixels[2]
    bottom_left = rect_pixels[3]

    origin_y = min(top_left.y, top_right.y)
    max_y = max(bottom_left.y, bottom_right.y)
    height = max_y - origin_y

    origin_x = min(top_left.x, bottom_left.x)
    max_x = max(top_right.x, bottom_right.x)
    width = max_x - origin_x

    origin = Point(origin_x, origin_y)
    size = Size(width, height)

    rect = Rect(origin, size)

    origin_y = rect.origin.y
    origin_x = rect.origin.x
    width = rect.size.width
    height = rect.size.height

    mouth_image = image.copy()
    mouth_image = mouth_image[origin_y:origin_y + height, origin_x:origin_x + width]

    return mouth_image, rect.origin


def _face_angle(landmark, image_shape) -> float:
    # left_eye_points = convert_landmark_points_to_pixels(landmark, LEFT_EYE_INDEXES, image_shape)
    # right_eye_points = convert_landmark_points_to_pixels(landmark, RIGHT_EYE_INDEXES, image_shape)
    #
    # left_eye_center_x = left_eye_points[0].x + (left_eye_points[1].x - left_eye_points[0].x) / 2
    # left_eye_center_y = left_eye_points[2].y + (left_eye_points[3].y - left_eye_points[2].y) / 2
    #
    # right_eye_center_x = right_eye_points[0].x + (right_eye_points[1].x - right_eye_points[0].x) / 2
    # right_eye_center_y = right_eye_points[2].y + (right_eye_points[3].y - right_eye_points[2].y) / 2
    #
    # delta_y = right_eye_center_y - left_eye_center_y
    # delta_x = right_eye_center_x - left_eye_center_x

    points = convert_landmark_points_to_pixels(landmark, [130, 359], image_shape)

    left_point = points[0]
    right_point = points[1]

    delta_y = right_point.y - left_point.y
    delta_x = right_point.x - left_point.x

    return -(atan2(delta_y, delta_x) * 180 / pi)


def _blur_radius(image_size: Size) -> int:
    resolution = image_size.height * image_size.width

    if resolution < 3_360_000:
        return 1
    elif resolution in range(3_360_000, 18_100_000):
        return 3
    elif resolution in range(18_100_000, 20_100_00):
        return 6
    else:
        return 4


def _paste_teeth(original_image: Image,
                 teeth_image_type: TeethImageType,
                 teeth_polygon: list[Point],
                 landmark,
                 gum_info: GumData) -> Image:

    """Assemble a new image from the original and one with cropped mouth area"""
    image_shape = convert_pil_image_to_cv(original_image).shape

    nose_points = convert_landmark_points_to_pixels(landmark, NOSE_WIDTH_INDEXES, image_shape)
    interalar_distance = nose_points[1].x - nose_points[0].x

    teeth_image = teeth_image_type.image()

    ratio = (teeth_image_type.intercanine_distance() / interalar_distance) * 1.1
    teeth_width = int(teeth_image.size[0] / ratio)
    teeth_height = int(teeth_image.size[1] / ratio)

    resized_teeth = teeth_image.resize((teeth_width, teeth_height), Image.ANTIALIAS)

    angle = _face_angle(landmark, image_shape)
    rotated_teeth = resized_teeth.rotate(angle)

    gum_offset: int

    if gum_info.type == GumType.NONE:
        gum_offset = int(teeth_image_type.gum_none_height() / ratio)  # Come up with better value/logic than 5. Some value to put teeth in a way gum between teeth is not visible at all
    elif gum_info.type == GumType.SLIGHT:
        gum_offset = int((teeth_image_type.gum_full_height() / ratio) - gum_info.height)
    else:
        gum_offset = int((teeth_image_type.gum_above_teeth_height() / ratio) - gum_info.height)

    mouth_center = convert_landmark_points_to_pixels(landmark, MOUTH_CENTER_INDEX, image_shape)
    mouth_center = mouth_center[0]

    # origin_x = int(mouth_center.x - (resized_teeth.size[0] / 2))
    origin_x = int(mouth_center.x - (teeth_image_type.center_x() / ratio))

    top_mouth_point = min(teeth_polygon, key=lambda point: hypot(point.x - mouth_center.x, point.y - mouth_center.y))
    origin_y = top_mouth_point.y - gum_offset

    teeth_origin = (origin_x, origin_y)

    teeth_on_background = Image.new('RGBA', original_image.size, (0, 0, 0))
    teeth_on_background.paste(rotated_teeth, teeth_origin, rotated_teeth)

    mask = Image.new('L', original_image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(teeth_polygon, fill=255)

    blur_radius = _blur_radius(Size(image_shape[1], image_shape[0]))
    blur = mask.filter(ImageFilter.GaussianBlur(blur_radius))

    return Image.composite(teeth_on_background, original_image, blur)


def replace_teeth(image_path: str) -> None:
    """Main function to replace teeth"""

    # pass image directly, not path
    multi_face_landmarks = multi_face_landmarks_from_image(image_path)

    # what if multiple faces returned?
    if not multi_face_landmarks:
        print('No face found')
        return

    landmark = multi_face_landmarks[0].landmark
    image = cv2.imread(image_path)

    # crop mouth from face
    mouth_image, mouth_origin = _cropped_mouth_image(landmark, image)

    # use lip detector to get mouth mask
    mouth_image = convert_cv_image_to_pil(mouth_image)
    crop_mask = inner_mouth_mask_from_image(mouth_image)

    # get mouth contour using the mask
    crop_mask = convert_pil_image_to_cv(crop_mask)
    mouth_contour = _contour_from_mask(crop_mask)

    # calculate gum height
    mouth_top_center_points = convert_landmark_points_to_pixels(landmark, MOUTH_TOP_CENTER_INDEXES, image.shape)

    mouth_top_center_points = list(map(lambda point: Point(point.x - mouth_origin.x, point.y - mouth_origin.y),
                                       mouth_top_center_points))

    mouth_analyzer = MouthAnalyzer()
    mouth_analyzer.process(mouth_image.convert('RGBA'),
                           mouth_contour,
                           range(mouth_top_center_points[0].x, mouth_top_center_points[1].x))

    # Determine if jaw is narrow
    # left_range_points = convert_landmark_points_to_pixels(landmark, [78, 91], image.shape)
    # left_range_points = list(map(lambda point: Point(point.x - mouth_origin.x, point.y - mouth_origin.y),
    #                              left_range_points))
    # left_range = range(left_range_points[0].x, left_range_points[1].x)
    #
    # right_range_points = convert_landmark_points_to_pixels(landmark, [415, 308], image.shape)
    # right_range_points = list(map(lambda point: Point(point.x - mouth_origin.x, point.y - mouth_origin.y),
    #                               right_range_points))
    # right_range = range(right_range_points[0].x, right_range_points[1].x)

    # is_narrow_jaw = mouth_analyzer.is_narrow_jaw(left_range, right_range)

    gum_info = mouth_analyzer.gum_data()
    teeth_color = mouth_analyzer.teeth_color()
    is_opened_mouth = mouth_analyzer.is_opened_mouth()
    teeth_image_type = TeethImageProvider.image_type(teeth_color, is_opened=is_opened_mouth)

    # map contour to original face image coordinates
    contour = list(map(lambda point: Point(point.x + mouth_origin.x, point.y + mouth_origin.y), mouth_contour))

    image = Image.open(image_path).convert('RGBA')
    result_image = _paste_teeth(image, teeth_image_type, contour, landmark, gum_info)
    result_image.show()

    # os.makedirs(os.path.dirname(_OUTPUT_IMAGE_PATH), exist_ok=True)
    # result_image.save(_OUTPUT_IMAGE_PATH, 'PNG')
