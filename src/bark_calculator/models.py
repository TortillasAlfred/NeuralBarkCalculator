from torch import nn
from torchvision.models import vgg19_bn
import math


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


class RegressionVGG19_BN(nn.Module):

    def __init__(self):
        super().__init__()
        model = vgg19_bn(pretrained=True)

        for params in model.parameters():
            params.requires_grad = False

        regressor = nn.Sequential(
            nn.Linear(512 * 7 * 7, 50176),
            nn.Sigmoid()
        )
        regressor.apply(initialize_weights)
        model.classifier = regressor
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x.reshape(x.size(0), 224, 224).unsqueeze(1)
