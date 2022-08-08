import cv2
import numpy as np


def get_geometric_centroid(contour):
    x = contour.transpose()[0][0].mean()
    y = contour.transpose()[1][0].mean()
    return x, y


def geometric_filter(mask, depth_thresh=2000, min_length_thresh=100, off_center_thresh=40):
    """
    to reject more images:

    depth can go lower (eg to ~1000-1500)
    min_length can go higher (eg to ~150-200)
    off_center_thresh can go lower (eg to ~25-30)

    and reversely for allowing more images through
    """

    mask = mask.astype(np.uint8)

    connected_components = cv2.connectedComponentsWithStats(mask)
    if connected_components[0] != 2:
        return False

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 1:
        return False

    else:
        contour = contours[0]

        hull = cv2.convexHull(contour, returnPoints=False)

        defects = cv2.convexityDefects(contour, hull)

        cent_x, cent_y = get_geometric_centroid(contour)

        for i in range(defects.shape[0]):
            start_point, end_point, farthest_point, depth = defects[i, 0]
            if depth > depth_thresh:
                defect_x, defect_y = contours[farthest_point][0]
                if (abs(start_point - end_point) > min_length_thresh) and (abs(defect_x - cent_x) < off_center_thresh):
                    continue
                else:
                    return False

    return True


def confidence_filter(mask_conf_levels, thresh=0.5, min_conf_above_thresh=0.96, min_ratio=0.6):
    """
    the thresh performs best at 0.5 in the tests done so far.

    to reject more images:

    min_conf_above_thresh can go higher (eg to ~.98-.99)
    and min_ratio can go higher (eg to ~.75-.85)

    and reversely for allowing more images through
    """

    threshed_mask = mask_conf_levels[mask_conf_levels >= thresh]
    if np.mean(threshed_mask) < min_conf_above_thresh:
        return False

    high_conf_thresh = 0.9999
    high_conf_mask = mask_conf_levels[mask_conf_levels >= high_conf_thresh]

    if high_conf_mask.shape[0] / threshed_mask.shape[0] < min_ratio:
        return False

    return True
