import pandas as pd
import json
import os
import cv2
import numpy as np

faces_dir = 'C:/projects/lips/dataset_1k_test/images'
save_masks_pth = 'C:/projects/lips/dataset_1k_test/masks'

labels = pd.read_excel('C:/projects/lips/dataset_1k_test/dataset.xlsx', header=1)
# labels = pd.read_csv('C:/projects/lips/1k_dataset/dataset.csv', delimiter=',')

# labels.drop('Unnamed: 0', axis=1, inplace=True)


def load_coords(j):
    return json.loads(j)['all_points_x'], json.loads(j)['all_points_y']


def draw_coords(fname, coords, idx):
    xs = coords[1]
    ys = coords[0]
    im_path = os.path.join(faces_dir, fname)
    im = cv2.imread(im_path)
    for x, y in zip(xs, ys):
        im[x, y, :] = 0

    cv2.imwrite(os.path.join(save_masks_pth, fname), im)


for i in range(len(labels)):
    fname = labels.iloc[i, 0]
    im_path = os.path.join(faces_dir, fname)
    im = cv2.imread(im_path)

#     coords = load_coords(labels.iloc[i, -2])
    coords = [[x, y] for x, y in zip(json.loads(labels.iloc[i, 1]), json.loads(labels.iloc[i, 2]))]

    pts = np.array(coords)
    new_im = np.zeros(im.shape)

    new_im = cv2.fillPoly(new_im, [pts], (255, 255, 255))
    cv2.imwrite(os.path.join(save_masks_pth, os.path.splitext(fname)[0] + '.png'), new_im)