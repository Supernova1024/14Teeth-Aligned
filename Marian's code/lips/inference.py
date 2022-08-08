import json
import torch

from PIL import Image
from scipy.special import expit

from torch import nn
from torchvision import transforms, models

from post_filters import geometric_filter, confidence_filter


class SegModel(nn.Module):
    def __init__(self, net):
        super(SegModel, self).__init__()
        self.net = net

    def forward(self, x):
        y = self.net(x)['out']
        return y


class Segmentator:
    def __init__(self, ckpt_path, norms_path):
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        self.device = torch.device(dev)

        with open(norms_path, 'r') as f:
            norms = json.load(f)
            self.ds_mean = norms['mean']
            self.ds_std = norms['std']

        self.preprocess = transforms.Compose([transforms.Resize((288, 384)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=self.ds_mean, std=self.ds_std)])

        net = models.segmentation.deeplabv3_resnet101(num_classes=2)

        self.checkpoint = torch.load(ckpt_path, map_location=torch.device(self.device))
        self.model = SegModel(net=net)
        self.model.load_state_dict(self.checkpoint['state_dict'])
        self.model.eval()
        self.model.to(self.device)

    def predict(self, img_path, thresh=0.5):
        assert 0 < thresh < 1, 'threshold must be in 0 to 1 range'
        mask = self.inference(img_path)

        mask[mask < thresh] = 0
        mask[mask >= thresh] = 1

        return mask

    def inference(self, img_path):
        img = Image.open(img_path)
        x = self.preprocess(img)
        with torch.no_grad():
            x = x.to(self.device).float()
            y = self.model(x.unsqueeze(0).float())
            ypos = y[0, 1, :, :].clone().detach().cpu().numpy()
            mask = expit(ypos)
        return mask


def resize_mask_to_img_orig_size(img_path, mask):
    img = Image.open(img_path)
    orig_img_sz = img.size
    mask = mask.astype('float32')
    mask = mask * 255
    mask_img = Image.fromarray(mask)
    mask_img = mask_img.convert('RGB')
    mask_orig_sz = mask_img.resize(orig_img_sz)
    return mask_orig_sz


if __name__ == '__main__':
    norms_path = 'dataset_1k/mean_std.json'
    ckpt_path = 'checkpoints/fromscratch_1k_deeplab101_0.0001_0.95_50.pt'
    test_img_path = 'dataset_mouth/images/smile00001.jpg'

    seg = Segmentator(ckpt_path, norms_path)
    mask = seg.predict(test_img_path, thresh=0.8)
    mask_conf_levels = seg.inference(test_img_path)

    if geometric_filter(mask) and confidence_filter(mask_conf_levels):
        print('passed')


