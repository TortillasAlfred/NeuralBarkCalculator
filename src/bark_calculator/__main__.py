from dataset import RegressionDatasetFolder, make_weight_map
from utils import *
from models import vanilla_unet

from torchvision.transforms import *

from poutyne.framework import Experiment
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.nn.modules.loss import BCEWithLogitsLoss
import torch

import numpy as np


if __name__ == "__main__":
    mean, std = get_mean_std()
    dataset = RegressionDatasetFolder("/mnt/storage/mgodbout/Ecorcage/Images/nn",
                                      input_only_transform=Compose(
                                          [Normalize(mean, std)]
                                      ),
                                      transform=Compose(
                                          [Resize(256), ToTensor()]
                                      ))
    augmented_dataset = RegressionDatasetFolder("/mnt/storage/mgodbout/Ecorcage/Images/nn",
                                                input_only_transform=Compose(
                                                    [Normalize(mean, std),
                                                     ToPILImage(),
                                                     ColorJitter(brightness=0.05,
                                                                 contrast=0.05,
                                                                 saturation=0.05,
                                                                 hue=0.05),
                                                     ToTensor()]
                                                ),
                                                transform=Compose(
                                                    [RandomRotation(180, expand=True),
                                                     RandomResizedCrop(256),
                                                     ToTensor()]
                                                ))
    # for sample in iter(dataset):
    #     target = sample[1]
    #     target_name = sample[2].split('/')[-1]
    #     target_name = target_name.replace(".bmp", ".npy")
    #     target_weight = make_weight_map(target.numpy())
    #     np.save("/mnt/storage/mgodbout/Ecorcage/Images/nn/target_weights/" + target_name, target_weight)

    # for sample, augmented_sample in zip(iter(dataset), iter(augmented_dataset)):
    #     _,  axs = plt.subplots(2, 2)

    #     sample += augmented_sample

    #     for ax, img in zip(axs.flatten(), sample):
    #         ax.imshow(img)
    #         ax.axis('off')

    #     figManager = plt.get_current_fig_manager()
    #     figManager.window.showMaximized()
    #     plt.tight_layout()
    #     plt.show()

    train_sampler, valid_sampler = get_train_valid_samplers(dataset,
                                                            train_percent=0.8)
    train_loader = DataLoader(dataset, batch_size=1,
                              sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=1,
                              sampler=valid_sampler)
    exp = Experiment(directory="/mnt/storage/mgodbout/Ecorcage/raw_unet/",
                     module=vanilla_unet(),
                     device=torch.device("cuda:0"),
                     optimizer="adam",
                     type="reg",
                     loss_function=MixedLoss())

    exp.train(train_loader=train_loader,
              valid_loader=valid_loader,
              epochs=1000)
