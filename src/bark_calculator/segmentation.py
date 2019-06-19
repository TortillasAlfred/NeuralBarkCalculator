import torchvision
from torchvision.models.segmentation._utils import _SimpleSegmentationModel
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.detection.backbone_utils import IntermediateLayerGetter
from torchvision.models import resnet

import PIL
import torch


def deeplabv3_resnet101():
    backbone = resnet.__dict__['resnet101'](
        pretrained=False, replace_stride_with_dilation=[False, True, True])

    return_layers = {'layer4': 'out'}

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    inplanes = 2048
    classifier = DeepLabHead(inplanes, 3)

    return _SimpleSegmentationModel(backbone, classifier)


module = deeplabv3_resnet101()
# set it to evaluation mode, as the model behaves differently
# during training and during evaluation
# model.eval()
module.to('cuda:0')
module.eval()

with torch.no_grad():
    image = PIL.Image.open(
        '/home/magod/Documents/Encorcage/Images/dual_exp/samples/145.bmp')
    image = torchvision.transforms.functional.resize(image, [2048, 2048])
    image = torchvision.transforms.functional.to_tensor(image)
    image = image.unsqueeze(0).to('cuda:0')

    # pass a list of (potentially different sized) tensors
    # to the model, in 0-1 range. The model will take care of
    # batching them together and normalizing
    # output = model([image_tensor])
    output_seg = module(image)
    # output is a list of dict, containing the postprocessed
    # predictions

    print('eh')
