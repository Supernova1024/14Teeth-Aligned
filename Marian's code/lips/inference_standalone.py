from PIL import Image
import torch
from torch import nn
import json
from scipy.special import expit

from torchvision import transforms, models


class SegModel(nn.Module):
    def __init__(self, net):
        super(SegModel, self).__init__()
        self.net = net

    def forward(self, x):
        y = self.net(x)['out']
        return y


def run_inference(img_path, thresh=0.5):
    img = Image.open(img_path)
    Xtest = preprocess(img)

    with torch.no_grad():
        Xtest = Xtest.to(device).float()
        ytest = model(Xtest.unsqueeze(0).float())
        ypos = ytest[0, 1, :, :].clone().detach().cpu().numpy()

        sig_ypos = expit(ypos)
        sig_ypos[sig_ypos < thresh] = 0
        sig_ypos[sig_ypos >= thresh] = 1

    return sig_ypos


if __name__ == '__main__':
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    img_paths = ['dataset_mouth/images/smile00001.jpg', 'dataset_mouth/images/smile00002.jpg']
    ckpt_path = 'checkpoints/fromscratch_1k_deeplab101_0.0001_0.95_50.pt'
    norms_path = 'dataset_1k/mean_std.json'

    with open(norms_path, 'r') as f:
        norms = json.load(f)
        ds_mean = norms['mean']
        ds_std = norms['std']

    preprocess = transforms.Compose([transforms.Resize((288, 384)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=ds_mean, std=ds_std)])

    net = models.segmentation.deeplabv3_resnet101(num_classes=2)

    checkpoint = torch.load(ckpt_path, map_location=torch.device(dev))
    model = SegModel(net=net)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.to(device)

    for img_path in img_paths:
        orig_img_sz = Image.open(img_path).size
        mask = run_inference(img_path)
        mask = mask.astype('float32')
        mask = mask * 255
        mask_img = Image.fromarray(mask)
        mask_img = mask_img.convert('RGB')
        mask_orig_sz = mask_img.resize(orig_img_sz)