from dataset import RegressionDatasetFolder, make_weight_map, pil_loader
from utils import *
from models import fcn_resnet50

from torchvision.transforms import *

from poutyne.framework import Experiment, ExponentialLR, EarlyStopping
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
    levels = [('combined_images', ['train', 'valid', 'test']), ('outputs', ['train', 'valid', 'test'])]

    results_dir = os.path.join(root_dir, 'Images', 'results', 'ng_2')

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
    test_dataset = RegressionDatasetFolder(os.path.join(args.root_dir, 'Images/dual_exp'),
                                           input_only_transform=Compose([Normalize(mean, std)]),
                                           transform=Compose([ToTensor()]))

    valid_dataset = RegressionDatasetFolder(os.path.join(args.root_dir, 'Images/dual_exp'),
                                            input_only_transform=Compose([Normalize(mean, std)]),
                                            transform=Compose([ToTensor()]),
                                            include_fname=True)

    module = fcn_resnet50()

    optim = torch.optim.Adam(module.parameters(), lr=1e-3, weight_decay=5e-3)
    exp = Experiment(directory=os.path.join(args.root_dir, 'ng_2/'),
                     module=module,
                     device=torch.device(args.device),
                     optimizer=optim,
                     loss_function=CustomWeightedCrossEntropy(torch.tensor(pos_weights).to(args.device)),
                     metrics=[IOU(None)],
                     monitor_metric='val_IntersectionOverUnion',
                     monitor_mode='max')

    lr_schedulers = [ExponentialLR(gamma=0.975)]
    callbacks = []

    pure_dataset = RegressionDatasetFolder(os.path.join(args.root_dir, 'Images/dual_exp'),
                                           input_only_transform=None,
                                           transform=Compose([ToTensor()]),
                                           include_fname=True)

    pure_loader = DataLoader(pure_dataset, batch_size=1, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=4, pin_memory=True)

    exp.load_best_checkpoint()
    module = exp.model.model
    module.eval()

    generate_output_folders(args.root_dir)

    results_csv = [[
        'Name', 'Type', 'F1_nothing', 'F1_bark', 'F1_node', 'F1_mean', 'Output Bark %', 'Output Node %',
        'Target Bark %', 'Target Node %'
    ]]

    with torch.no_grad():
        for image_number, (batch, pure_batch) in enumerate(zip(valid_loader, pure_loader)):
            input = pure_batch[0]
            target = pure_batch[1]
            fname = pure_batch[2][0]
            wood_type = pure_batch[3][0]

            del pure_batch

            # if os.path.isfile('/mnt/storage/mgodbout/Ecorcage/Images/results/ng_2/{}'.format(fname)):
            #     continue

            outputs = module(batch[0].to(torch.device(args.device)))
            outputs = torch.argmax(outputs, dim=1)
            outputs = remove_small_zones(outputs)

            del batch

            names = ['Input', 'Target', 'Generated image']

            imgs = [input, target, outputs]
            imgs = [img.detach().cpu().squeeze().numpy() for img in imgs]

            try:
                class_accs = f1_score(imgs[1].flatten(), imgs[2].flatten(), labels=[0, 1, 2], average=None)
                acc = class_accs.mean()
            except ValueError:
                print('Error on file {}'.format(fname))
                print(imgs[1].shape)
                print(imgs[2].shape)
                continue

            _, axs = plt.subplots(1, 3)

            for i, ax in enumerate(axs.flatten()):
                img = imgs[i]

                if len(img.shape) == 3:
                    img = img.transpose(1, 2, 0)

                ax.imshow(img)
                ax.set_title(names[i])
                ax.axis('off')

            suptitle = 'Mean f1 : {:.3f}'.format(acc)

            running_csv_stats = [fname, wood_type]

            class_names = ['Nothing', 'Bark', 'Node']

            for c, c_acc in zip(class_names, class_accs):
                suptitle += '\n{} : {:.3f}'.format(c, c_acc)
                running_csv_stats.append('{:.3f}'.format(c_acc))

            running_csv_stats.append('{:.3f}'.format(acc))

            for class_idx in [1, 2]:
                class_percent = (outputs == class_idx).float().mean().cpu()
                running_csv_stats.append('{:.5f}'.format(class_percent))

            for class_idx in [1, 2]:
                class_percent = (target == class_idx).float().mean().cpu()
                running_csv_stats.append('{:.5f}'.format(class_percent))

            plt.suptitle(suptitle)
            plt.tight_layout()
            # plt.show()
            plt.savefig(os.path.join(args.root_dir,
                                     'Images/results/ng_2/combined_images/{}/{}').format(wood_type, fname),
                        format='png',
                        dpi=900)
            plt.close()

            outputs = outputs.squeeze().cpu().numpy()
            dual_outputs = np.zeros((outputs.shape[0], outputs.shape[1]), dtype=np.uint8)
            dual_outputs[outputs == 1] = 127
            dual_outputs[outputs == 2] = 255

            dual = Image.fromarray(dual_outputs, mode='L')
            dual.save(os.path.join(args.root_dir, 'Images/results/ng_2/outputs/{}/{}').format(wood_type, fname))

            results_csv.append(running_csv_stats)

    csv_file = os.path.join(args.root_dir, 'Images', 'results', 'ng_2', 'final_stats.csv')

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
