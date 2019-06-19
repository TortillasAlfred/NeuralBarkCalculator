from dataset import RegressionDatasetFolder, make_weight_map, pil_loader
from utils import *
from models import vanilla_unet, FCDenseNet103, FCDenseNet57, B2B, deeplabv3_resnet101, fcn_resnet50

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


def main(args):
    mean, std = get_mean_std()
    pos_weights = get_pos_weight()
    test_dataset = RegressionDatasetFolder(
        os.path.join(args.root_dir, "Images/dual_exp"),
        input_only_transform=Compose([Normalize(mean, std)]),
        transform=Compose([ToTensor()]))

    module = fcn_resnet50()

    optim = torch.optim.Adam(module.parameters(), lr=1e-3, weight_decay=5e-3)
    exp = Experiment(directory=os.path.join(args.root_dir, "best_model/"),
                     module=module,
                     device=torch.device("cuda:1"),
                     optimizer=optim,
                     loss_function=CustomWeightedCrossEntropy(
                         torch.tensor(pos_weights).to('cuda:1')),
                     metrics=[IOU(None)],
                     monitor_metric='val_IntersectionOverUnion',
                     monitor_mode='max')

    lr_schedulers = [ExponentialLR(gamma=0.975)]
    callbacks = []

    all_dataset = RegressionDatasetFolder(
        os.path.join(args.root_dir, "Images/dual_exp"),
        input_only_transform=Compose([Normalize(mean, std)]),
        transform=Compose([ToTensor()]),
        include_fname=True)

    all_loader = DataLoader(all_dataset, batch_size=1, num_workers=4)

    module = exp.model.model
    module.eval()

    if not os.path.isdir(
            "/mnt/storage/mgodbout/Ecorcage/Images/results/best_model"):
        os.makedirs("/mnt/storage/mgodbout/Ecorcage/Images/results/best_model")

    if not os.path.isdir(
            "/mnt/storage/mgodbout/Ecorcage/Images/results/best_model/train"):
        os.makedirs(
            "/mnt/storage/mgodbout/Ecorcage/Images/results/best_model/train")

    if not os.path.isdir(
            "/mnt/storage/mgodbout/Ecorcage/Images/results/best_model/valid"):
        os.makedirs(
            "/mnt/storage/mgodbout/Ecorcage/Images/results/best_model/valid")

    if not os.path.isdir(
            "/mnt/storage/mgodbout/Ecorcage/Images/results/best_model/test"):
        os.makedirs(
            "/mnt/storage/mgodbout/Ecorcage/Images/results/best_model/test")

    splits = [(train_split, 'train'), (valid_split, 'valid'),
              (test_split, 'test')]

    with torch.no_grad():
        for image_number, (batch, pure_batch) in enumerate(
                zip(valid_loader, pure_loader)):
            input = pure_batch[0]
            target = pure_batch[1]
            fname = pure_batch[2][0]

            del pure_batch

            # if os.path.isfile("/mnt/storage/mgodbout/Ecorcage/Images/results/best_model/{}".format(fname)):
            #     continue

            outputs = module(batch[0].to(torch.device("cuda:1")))
            outputs = torch.argmax(outputs, dim=1)
            outputs = remove_small_zones(outputs)

            del batch

            names = ["Input", "Target", "Generated image"]

            imgs = [input, target, outputs]
            imgs = [img.detach().cpu().squeeze().numpy() for img in imgs]

            try:
                class_accs = f1_score(imgs[1].flatten(),
                                      imgs[2].flatten(),
                                      labels=[0, 1, 2],
                                      average=None)
                acc = class_accs.mean()
            except ValueError:
                print("Error on file {}".format(fname))
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

            suptitle = "Mean f1 : {:.3f}".format(acc)

            class_names = ["Nothing", "Bark", "Node"]

            for c, c_acc in zip(class_names, class_accs):
                suptitle += "\n{} : {:.3f}".format(c, c_acc)

            for split_idxs, split_name in splits:
                if image_number in split_idxs:
                    split = split_name

            plt.suptitle(suptitle)
            plt.tight_layout()
            # plt.show()
            plt.savefig(
                "/mnt/storage/mgodbout/Ecorcage/Images/results/best_model/{}/{}"
                .format(split, fname),
                format="png",
                dpi=900)
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('root_dir', type=str, help='root directory path.')

    parser.add_argument('model_save_dir',
                        type=str,
                        help='root where computed data stats will be saved.')

    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='Which torch device to train with.',
                        choices=['cpu', 'cuda:0', 'cuda:1'])

    args = parser.parse_args()

    main()
