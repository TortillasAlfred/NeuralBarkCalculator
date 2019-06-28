import torch
import torch.nn as nn
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models.detection.backbone_utils import IntermediateLayerGetter
from torchvision.models import resnet
from torchvision.transforms import ToTensor
from os.path import join
import numpy as np
from skimage.transform import resize
from skimage.io import imsave
from dataset import RegressionDatasetFolder
from tqdm import tqdm
import warnings


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


def trim_black(image):
    summed_image = np.sum(image, axis=-1)
    summed_image = summed_image > 1e-3

    clear_enough_lines_idx = np.mean(summed_image, axis=-1) > 0.85

    first_idx = np.argmax(clear_enough_lines_idx)
    last_idx = image.shape[0] - np.argmax(clear_enough_lines_idx[::-1])

    return image[first_idx:last_idx]


class NeuralBarkCalculator():

    DEFAULT_MEAN = [0.7399, 0.6139, 0.4401]
    DEFAULT_STD = [0.1068, 0.1272, 0.1271]

    def __init__(self, model_path, mean=DEFAULT_MEAN, std=DEFAULT_STD, target_size=1024):
        super().__init__()
        self.model = fcn_resnet50()
        self.model.load_state_dict(torch.load(model_path))
        self.mean = mean
        self.std = std
        self.target_size = 1024

    def to(self, device):
        self.model.to(device)

    def predict(self, root_path):
        output_path = join(root_path, 'processed')
        dataset = self._preprocess_images(root_path, output_path)

        output_path = join(root_path, 'results')
        self._predict_images(dataset, output_path)

    def _preprocess_images(self, root_path, output_path):
        raw_dataset = RegressionDatasetFolder(root_path, input_only_transform=ToTensor(), include_fname=True)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for img, _, fname, wood_type in tqdm(iter(raw_dataset),
                                                 total=len(raw_dataset),
                                                 ascii=True,
                                                 desc='Preprocessing images'):
                fname = str.replace(fname, '.bmp', '.png')
                img_output_path = join(output_path, 'samples', wood_type, fname)

                self._preprocess_image(img, img_output_path)

    def _preprocess_image(self, image, output_path):
        image = image.detach().cpu().numpy().transpose(1, 2, 0)

        if max(image.shape) > 1024:
            image = resize(image, (1024, 1024), order=3, mode='reflect', anti_aliasing=False)

        image = trim_black(image)

        imsave(output_path, image)

    def _predict_images(self, dataset, output_path):
        pass
