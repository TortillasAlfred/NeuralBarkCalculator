from utils import remove_small_zones
from dataset import RegressionDatasetFolder

import torch
import torch.nn as nn
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models.detection.backbone_utils import IntermediateLayerGetter
from torchvision.models import resnet
from torchvision.transforms import ToTensor, Normalize
from torch.utils.data import DataLoader
from os.path import join
import numpy as np
from skimage.transform import resize
from skimage.io import imsave
from tqdm import tqdm
import warnings

import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import torch
from PIL import Image
import csv


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


def fcn_resnet50(pretrained=True):
    backbone = resnet.__dict__['resnet50'](pretrained=pretrained, replace_stride_with_dilation=[False, True, True])

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
    DEFAULT_MM_PER_PIXEL = 0.9 * 0.9

    def __init__(self,
                 model_path,
                 device,
                 mean=DEFAULT_MEAN,
                 std=DEFAULT_STD,
                 target_size=1024,
                 mm_per_pix=DEFAULT_MM_PER_PIXEL):
        super().__init__()
        self.device = device
        self.model = fcn_resnet50(pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.mean = mean
        self.std = std
        self.target_size = 1024
        self.mm_per_pix = mm_per_pix

    def predict(self, root_path):
        processed_path = join(root_path, 'processed')
        dataset = self._preprocess_images(root_path, processed_path)

        output_path = join(root_path, 'results')
        valid_dataset = RegressionDatasetFolder(processed_path,
                                                input_only_transform=Normalize(self.mean, self.std),
                                                transform=ToTensor(),
                                                include_fname=True)

        pure_dataset = RegressionDatasetFolder(processed_path,
                                               input_only_transform=None,
                                               transform=ToTensor(),
                                               include_fname=True)

        self._predict_images(valid_dataset, pure_dataset, output_path)

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

        if image.shape[0] == image.shape[1]:  #Untrimmed
            image = trim_black(image)

        imsave(output_path, image)

    def _predict_images(self, valid_dataset, pure_dataset, output_path):
        pure_loader = DataLoader(pure_dataset, batch_size=1)
        valid_loader = DataLoader(valid_dataset, batch_size=1)

        results_csv = [[
            'Name', 'Type', 'Image Size', 'Output Bark %', 'Bark area (mm^2)', 'Output Node %', 'Node area (mm^2)'
        ]]

        with torch.no_grad():
            for image_number, (batch, pure_batch) in tqdm(enumerate(zip(valid_loader, pure_loader)),
                                                          total=len(pure_loader),
                                                          ascii=True,
                                                          desc='Predicted images'):
                input = pure_batch[0]
                fname = pure_batch[2][0]
                wood_type = pure_batch[3][0]

                del pure_batch

                outputs = self.model(batch[0].to(self.device))
                outputs = torch.argmax(outputs, dim=1)
                outputs = remove_small_zones(outputs)

                del batch

                names = ['Input', 'Generated image']

                imgs = [input, outputs]
                imgs = [img.detach().cpu().squeeze().numpy() for img in imgs]

                _, axs = plt.subplots(1, 2)

                for i, ax in enumerate(axs.flatten()):
                    img = imgs[i]

                    if len(img.shape) == 3:
                        img = img.transpose(1, 2, 0)

                    ax.imshow(img)
                    ax.set_title(names[i])
                    ax.axis('off')

                img_size = '{} x {}'.format(img.shape[0], img.shape[1])

                running_csv_stats = [fname, wood_type, img_size]

                class_names = ['Nothing', 'Bark', 'Node']
                class_percents = []

                for class_idx in [1, 2]:
                    n_pixels = (outputs == class_idx).float().cpu()

                    class_percent = n_pixels.mean()
                    class_percents.append(class_percent * 100)
                    running_csv_stats.append('{:.5f}'.format(class_percent * 100))

                    class_area = (n_pixels.sum() * self.mm_per_pix).item()
                    running_csv_stats.append('{:.5f}'.format(class_area))

                suptitle = 'Estimated composition percentages\n'

                for class_name, class_percent in zip(class_names[1:], class_percents):
                    suptitle += '{} : {:.3f}\n'.format(class_name, class_percent)

                plt.suptitle(suptitle)
                plt.tight_layout()
                plt.savefig(join(output_path, 'combined_images', wood_type, fname), format='png', dpi=900)
                plt.close()

                outputs = outputs.squeeze().cpu().numpy()
                dual_outputs = np.zeros((outputs.shape[0], outputs.shape[1]), dtype=np.uint8)
                dual_outputs[outputs == 1] = 127
                dual_outputs[outputs == 2] = 255

                dual = Image.fromarray(dual_outputs, mode='L')
                dual.save(join(output_path, 'outputs', wood_type, fname))

                results_csv.append(running_csv_stats)

        csv_file = join(output_path, 'final_stats.csv')

        with open(csv_file, 'w') as f:
            csv_writer = csv.writer(f, delimiter='\t')
            csv_writer.writerows(results_csv)
