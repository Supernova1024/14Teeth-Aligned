from torch import nn


class SegModel(nn.Module):
    def __init__(self, net):
        super(SegModel, self).__init__()
        self.net = net

    def forward(self, x):
        y = self.net(x)['out']
        return y