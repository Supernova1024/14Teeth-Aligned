import cv2
import numpy as np


def geometric_filter(mask, depth_thresh=1000, length_thresh=100, concave_length_thresh=2, concave_depth_thresh=1):
    # connected components and contours might be overlapping at 100% of the cases
    mask = mask.astype(np.uint8)

    connected_components = cv2.connectedComponentsWithStats(mask)
    if connected_components[0] != 2:
        return False

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 1:
        return False

    else:
        hull = cv2.convexHull(contours[0], returnPoints=False)

        defects = cv2.convexityDefects(contours[0], hull)

        concave_depth_count = 0
        concave_length_count = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            if d > depth_thresh:
                concave_depth_count += 1
            if abs(s - e) > length_thresh:
                concave_length_count += 1

        if (concave_depth_count > concave_depth_thresh) or (concave_length_count > concave_length_thresh):
            return False

    return True


def confidence_filter(mask_conf_levels, thresh=0.5, min_conf_above_thresh=0.92, min_ratio=0.5):
    threshed_mask = mask_conf_levels[mask_conf_levels >= thresh]
    if np.mean(threshed_mask) < min_conf_above_thresh:
        return False

    high_conf_thresh = 0.9999
    high_conf_mask = mask_conf_levels[mask_conf_levels >= high_conf_thresh]

    if high_conf_mask.shape[0] / threshed_mask.shape[0] < min_ratio:
        return False

    return True
