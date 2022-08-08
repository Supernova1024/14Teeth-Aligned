import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import F

from PIL import Image

from net_params import RE_SIZE


class SegDataset(Dataset):

    def __init__(self, images_list, masks_list, tfms_img, tfms_geo):
        self.images_list = images_list
        self.masks_list = masks_list
        self.tfms_img = tfms_img
        self.tfms_geo = tfms_geo

    def __getitem__(self, index):
        X = Image.open(self.images_list[index]).convert('RGB')
        y_img = Image.open(self.masks_list[index]).convert('RGB')

        X = F.to_tensor(X)
        y_img = F.to_tensor(y_img)
        X = F.resize(X, RE_SIZE)
        y_img = F.resize(y_img, RE_SIZE, interpolation=InterpolationMode.NEAREST)

        stacked = torch.cat([X, y_img], dim=0)

        if self.tfms_geo is not None:
            stacked = self.tfms_geo(stacked)

        X, y_img = torch.chunk(stacked, chunks=2, dim=0)

        X = self.tfms_img(X)

        y_img = F.rgb_to_grayscale(y_img)

        y1 = y_img.type(torch.BoolTensor)
        y2 = torch.bitwise_not(y1)
        y = torch.cat([y2, y1], dim=0)

        return X, y

    def __len__(self):
        return len(self.images_list)