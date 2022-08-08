from PIL import Image
import torch
from torch import nn
from scipy.special import expit
from torchvision import transforms, models
from script_location import script_location as location

checkpoint_path = f'{location}/checkpoints/checkpoint.pt'


class SegModel(nn.Module):
    def __init__(self, net):
        super(SegModel, self).__init__()
        self.net = net

    def forward(self, x):
        y = self.net(x)['out']
        return y


def run_inference(preprocess, device, model, image, thresh=0.5):
    Xtest = preprocess(image)

    with torch.no_grad():
        Xtest = Xtest.to(device).float()
        ytest = model(Xtest.unsqueeze(0).float())
        ypos = ytest[0, 1, :, :].clone().detach().cpu().numpy()
        yneg = ytest[0, 0, :, :].clone().detach().cpu().numpy()

        # ytest = ypos >= yneg

        sig_ypos = expit(ypos)
        sig_ypos[sig_ypos < thresh] = 0
        sig_ypos[sig_ypos >= thresh] = 1

    return sig_ypos


def inner_mouth_mask_from_image(image):
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    ds_mean = [0.6816867472863903, 0.4894418459102298, 0.4303826515845205]
    ds_std = [0.14059525858714297, 0.15456402696873536, 0.14602731700434263]

    preprocess = transforms.Compose([transforms.Resize((288, 384)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=ds_mean, std=ds_std)])

    net = models.segmentation.deeplabv3_resnet101(num_classes=2)

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model = SegModel(net=net)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    device = torch.device(dev)
    model.to(device)

    orig_img_sz = image.size
    mask = run_inference(preprocess, device, model, image)
    mask = mask.astype('float32')
    mask = mask * 255
    mask_img = Image.fromarray(mask)
    mask_img = mask_img.convert('RGBA')
    mask_orig_sz = mask_img.resize(orig_img_sz)
    return mask_orig_sz
