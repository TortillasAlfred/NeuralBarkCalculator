import torch
import torch.nn as nn
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models.detection.backbone_utils import IntermediateLayerGetter
from torchvision.models import resnet
from os.path import join
from poutyne.framework import Experiment


class SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]

        x = self.backbone(x)["out"]
        x = self.classifier(x)
        x = torch.nn.functional.interpolate(x, size=input_shape, mode='bicubic', align_corners=False)

        return x


def deeplabv3_resnet101():
    backbone = resnet.__dict__['resnet50'](pretrained=True, replace_stride_with_dilation=[False, True, True])

    return_layers = {'layer4': 'out'}

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    inplanes = 2048
    classifier = DeepLabHead(inplanes, 3)

    return SimpleSegmentationModel(backbone, classifier)


def fcn_resnet50():
    backbone = resnet.__dict__['resnet50'](pretrained=True, replace_stride_with_dilation=[False, True, True])

    return_layers = {'layer4': 'out'}

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    inplanes = 2048
    classifier = FCNHead(inplanes, 3)

    return SimpleSegmentationModel(backbone, classifier)


class NeuralBarkCalculator(nn.Module):

    DEFAULT_MEAN = [0.7399, 0.6139, 0.4401]
    DEFAULT_STD = [0.1068, 0.1272, 0.1271]

    def __init__(self, model_path, mean=DEFAULT_MEAN, std=DEFAULT_STD, target_size=1024):
        super().__init__()
        self.model = fcn_resnet50()
        self.model.load_state_dict(torch.load(model_path))
        self.mean = mean
        self.std = std
        self.target_size = 1024

    def predict(self, root_path):
        dataset = self._preprocess_images(root_path)

        output_path = join(root_path, 'results')
        self._predict_images(dataset, output_path)

    def _preprocess_images(self, root_path):
        pass

    def _predict_images(self, dataset, output_path):
        pass
