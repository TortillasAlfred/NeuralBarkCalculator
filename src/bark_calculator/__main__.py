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
                                          [ToTensor()]
                                      ))
    augmented_dataset = RegressionDatasetFolder("/mnt/storage/mgodbout/Ecorcage/Images/nn",
                                                input_only_transform=Compose(
                                                    [Normalize(mean, std)]
                                                ),
                                                transform=Compose(
                                                    [RandomRotation(180, expand=True),
                                                     RandomResizedCrop(1024),
                                                     ToTensor()]
                                                ))

    # for sample, augmented_sample in zip(iter(dataset), iter(augmented_dataset)):
    #     _,  axs = plt.subplots(2, 2)

    #     sample += augmented_sample

    #     for ax, img in zip(axs.flatten(), sample):
    #         img = ToPILImage()(img)
    #         ax.imshow(img)
    #         ax.axis('off')

    #     plt.tight_layout()
    #     plt.show()
    train_sampler, valid_sampler = get_train_valid_samplers(dataset,
                                                            train_percent=0.8)
    train_loader = DataLoader(augmented_dataset, batch_size=2,
                              sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=2,
                              sampler=valid_sampler)
    module = vanilla_unet()
    optim = torch.optim.Adam(module.parameters(), lr=1e-4)
    exp = Experiment(directory="/mnt/storage/mgodbout/Ecorcage/aug_unet/",
                     module=module,
                     device=torch.device("cuda:0"),
                     optimizer=optim,
                     type="reg",
                     loss_function=MixedLoss(),
                     metrics=[SoftDiceLoss()])

    exp.train(train_loader=train_loader,
              valid_loader=valid_loader,
              epochs=1000)
