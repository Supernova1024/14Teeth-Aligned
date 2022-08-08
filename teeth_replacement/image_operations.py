import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image, ImageDraw, ImageEnhance


def convert_cv_image_to_pil(cv_image):
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv_image)


def convert_pil_image_to_cv(pil_image):
    """Converts Pillow image to an OpenCV one"""
    rgb_image = pil_image.convert('RGB')
    cv_image = np.array(rgb_image)
    return cv_image[:, :, ::-1].copy()


def cropped_rgba_image(image, polygon):
    image_array = np.asarray(image)

    mask_image = Image.new('L', (image_array.shape[1], image_array.shape[0]), 0)
    ImageDraw.Draw(mask_image).polygon(polygon, outline=1, fill=1)
    mask = np.array(mask_image)

    new_image_array = np.empty(image_array.shape, dtype='uint8')
    new_image_array[:, :, :3] = image_array[:, :, :3]
    new_image_array[:, :, 3] = mask * 255

    return Image.fromarray(new_image_array, 'RGBA')


def cropped_binary_image(image, polygon):
    image_array = np.asarray(image)
    mask_image = Image.new('L', (image_array.shape[1], image_array.shape[0]), 0)

    ImageDraw.Draw(mask_image).polygon(polygon, outline=1, fill=1)
    mask = np.array(mask_image)

    new_image_array = np.empty(image_array.shape, dtype='uint8')
    new_image_array[:, :] = image_array[:, :] * mask

    return Image.fromarray(new_image_array, 'L')


def binary_image(image, alpha_channel=False):
    image = image.convert('LA' if alpha_channel else 'L')
    image = ImageEnhance.Contrast(image).enhance(3)

    threshold = 200
    return image.point(lambda pixel: 255 if pixel > threshold else 0)


def teeth_regions(mouth_image, mouth_polygon):
    binary = binary_image(mouth_image)
    cropped = cropped_binary_image(binary, mouth_polygon)
    smoothed = ndimage.gaussian_filter(cropped, 1)

    threshold = 100
    labeled, object_count = ndimage.label(smoothed > threshold)

    plt.imshow(labeled)
    plt.show()
