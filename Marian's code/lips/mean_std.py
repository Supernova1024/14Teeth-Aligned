import numpy as np
import cv2
import os
import json

from net_params import (
    DATASET_NAME,
    RE_SIZE
)


def get_mean_std():
    mean = np.zeros(3)
    std = np.zeros(3)
    imgs_path = os.path.join(DATASET_NAME, 'images')

    for img_file in os.listdir(imgs_path):
        image = cv2.imread(os.path.join(imgs_path, img_file))
        image = cv2.resize(image, RE_SIZE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255
        mean[0] += image[:, :, 0].mean()
        mean[1] += image[:, :, 1].mean()
        mean[2] += image[:, :, 2].mean()
        std[0] += image[:, :, 0].std()
        std[1] += image[:, :, 1].std()
        std[2] += image[:, :, 2].std()

    n_imgs = len(os.listdir(imgs_path))
    mean /= n_imgs
    std /= n_imgs

    with open(os.path.join(DATASET_NAME, 'mean_std.json'), 'w') as f:
        json.dump({'mean': list(mean), 'std': list(std)}, f)

    return mean, std


if __name__ == '__main__':
    get_mean_std()
