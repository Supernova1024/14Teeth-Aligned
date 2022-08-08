import os
import json

import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from mean_std import get_mean_std

from net_params import (
    DATASET_NAME,
    TRANSFORM_PARAMS as P
)


def get_norms(pth):
    with open(pth, 'r') as f:
        norms = json.load(f)
        mean = norms['mean']
        std = norms['std']
    return mean, std


norms_path = os.path.join(DATASET_NAME, 'mean_std.json')

if not os.path.exists(norms_path):
    get_mean_std()

DS_MEAN, DS_STD = get_norms(pth=norms_path)

TRANSFORMS = {
    'train': {
        'image': transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=P['brightness'],
                    contrast=P['contrast'],
                    saturation=P['saturation'],
                    hue=P['hue']
                ),
            ], p=P['p_random_apply']),
            # transforms.RandomApply([transforms.Lambda(lambda x: x + torch.randn_like(x))], p=P['p_noise']),
            transforms.Normalize(DS_MEAN, DS_STD)
        ]),
        'geo': transforms.Compose([
            transforms.RandomApply([
                transforms.RandomRotation(degrees=P['rotation'], interpolation=InterpolationMode.NEAREST),
                transforms.RandomAffine(degrees=P['affine'], interpolation=InterpolationMode.NEAREST),
                transforms.RandomPerspective(distortion_scale=P['perspective'], interpolation=InterpolationMode.NEAREST)
            ], p=P['p_random_apply']),
            transforms.RandomHorizontalFlip(p=P['p_flip'])
        ])
    },
    'val': {
        'image': transforms.Compose([
            transforms.Normalize(DS_MEAN, DS_STD)
        ])
    }
}