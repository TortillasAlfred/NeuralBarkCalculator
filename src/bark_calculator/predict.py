from dataset import RegressionDatasetFolder, pil_loader
from utils import *
from models import fcn_resnet50

from torchvision.transforms import *

from poutyne.framework import Experiment, ExponentialLR
from torch.utils.data import DataLoader, Subset, ConcatDataset
import matplotlib.pyplot as plt
from torch.nn.modules.loss import CrossEntropyLoss
from skimage.io import imread, imsave
import torch

from math import ceil
import numpy as np
import io
import pickle
from PIL import Image
import os
import argparse
import csv


def generate_output_folders(root_dir):
    wood_types = ["epinette_gelee", "epinette_non_gelee", "sapin"]
    levels = [('combined_images', []), ('outputs', [])]

    results_dir = os.path.join(root_dir, 'Images', 'results', 'all_3')

    def mkdirs_if_not_there(dir):
        if not os.path.isdir(dir):
            os.makedirs(dir)

    for folder, children in levels:
        current_dir = os.path.join(results_dir, folder)

        mkdirs_if_not_there(current_dir)

        for wood_type in wood_types:
            wood_dir = os.path.join(current_dir, wood_type)

            mkdirs_if_not_there(wood_dir)

            for child in children:
                child_dir = os.path.join(wood_dir, child)

                mkdirs_if_not_there(child_dir)


def main(args):
    mean, std = get_mean_std()
    pos_weights = get_pos_weight()
    test_dataset = RegressionDatasetFolder(os.path.join(args.root_dir, 'Images/1024_processed'),
                                           input_only_transform=Compose([Normalize(mean, std)]),
                                           transform=Compose([ToTensor()]))

    valid_dataset = RegressionDatasetFolder(os.path.join(args.root_dir, 'Images/1024_processed'),
                                            input_only_transform=Compose([Normalize(mean, std)]),
                                            transform=Compose([ToTensor()]),
                                            include_fname=True)

    module = fcn_resnet50()

    optim = torch.optim.Adam(module.parameters(), lr=1e-3, weight_decay=5e-3)
    exp = Experiment(directory=os.path.join(args.root_dir, 'all_3/'),
                     module=module,
                     device=torch.device(args.device),
                     optimizer=optim,
                     loss_function=CustomWeightedCrossEntropy(torch.tensor(pos_weights).to(args.device)),
                     metrics=[IOU(None)],
                     monitor_metric='val_IntersectionOverUnion',
                     monitor_mode='max')

    lr_schedulers = [ExponentialLR(gamma=0.975)]
    callbacks = []

    pure_dataset = RegressionDatasetFolder(os.path.join(args.root_dir, 'Images/1024_processed'),
                                           input_only_transform=None,
                                           transform=Compose([ToTensor()]),
                                           include_fname=True)

    pure_loader = DataLoader(pure_dataset, batch_size=1, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=4, pin_memory=True)

    exp.load_checkpoint(64)
    module = exp.model.model
    module.eval()

    generate_output_folders(args.root_dir)

    results_csv = [['Name', 'Type', 'Output Bark %', 'Output Node %']]

    with torch.no_grad():
        for image_number, (batch, pure_batch) in enumerate(zip(valid_loader, pure_loader)):
            input = pure_batch[0]
            fname = pure_batch[2][0]
            wood_type = pure_batch[3][0]

            del pure_batch

            # if os.path.isfile('/mnt/storage/mgodbout/Ecorcage/Images/results/all_3/{}'.format(fname)):
            #     continue

            outputs = module(batch[0].to(torch.device(args.device)))
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

            running_csv_stats = [fname, wood_type]

            class_names = ['Nothing', 'Bark', 'Node']
            class_percents = []

            for class_idx in [1, 2]:
                class_percent = (outputs == class_idx).float().mean().cpu()
                class_percents.append(class_percent * 100)
                running_csv_stats.append('{:.5f}'.format(class_percent * 100))

            suptitle = "Estimated composition percentages\n"

            for class_name, class_percent in zip(class_names[1:], class_percents):
                suptitle += "{} : {:.3f}\n".format(class_name, class_percent)

            plt.suptitle(suptitle)
            plt.tight_layout()
            # plt.show()
            plt.savefig(os.path.join(args.root_dir,
                                     'Images/results/all_3/combined_images/{}/{}').format(wood_type, fname),
                        format='png',
                        dpi=900)
            plt.close()

            outputs = outputs.squeeze().cpu().numpy()
            dual_outputs = np.zeros((outputs.shape[0], outputs.shape[1]), dtype=np.uint8)
            dual_outputs[outputs == 1] = 127
            dual_outputs[outputs == 2] = 255

            dual = Image.fromarray(dual_outputs, mode='L')
            dual.save(os.path.join(args.root_dir, 'Images/results/all_3/outputs/{}/{}').format(wood_type, fname))

            results_csv.append(running_csv_stats)

    csv_file = os.path.join(args.root_dir, 'Images', 'results', 'all_3', 'final_stats.csv')

    with open(csv_file, 'w') as f:
        csv_writer = csv.writer(f, delimiter='\t')
        csv_writer.writerows(results_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('root_dir', type=str, help='root directory path.')

    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='Which torch device to train with.',
                        choices=['cpu', 'cuda:0', 'cuda:1'])

    args = parser.parse_args()

    main(args)
