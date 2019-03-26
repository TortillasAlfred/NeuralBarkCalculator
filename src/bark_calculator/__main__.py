from dataset import RegressionDatasetFolder, make_weight_map
from utils import *
from models import vanilla_unet

from torchvision.transforms import *

from poutyne.framework import Experiment, ReduceLROnPlateau
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
                                          [Resize(256),
                                           RandomHorizontalFlip(),
                                           RandomVerticalFlip(),
                                           ToTensor()]
                                      ))
    augmented_dataset = RegressionDatasetFolder("/mnt/storage/mgodbout/Ecorcage/Images/nn",
                                                input_only_transform=Compose(
                                                    [Normalize(mean, std)]
                                                ),
                                                transform=Compose([RandomApply(
                                                    [RandomRotation(180, expand=False),
                                                     RandomResizedCrop(256)]
                                                ), Resize(256), ToTensor()]))

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
                                                            train_percent=0.6)
    train_loader = DataLoader(augmented_dataset, batch_size=4, shuffle=True)
    valid_loader = DataLoader(dataset, batch_size=4)
    module = vanilla_unet()
    optim = torch.optim.Adam(module.parameters(), lr=1e-4)
    exp = Experiment(directory="/mnt/storage/mgodbout/Ecorcage/mix_unet/",
                     module=module,
                     device=torch.device("cuda:0"),
                     optimizer=optim,
                     loss_function=MixedLoss())

    # exp.load_best_checkpoint()
    # module = exp.model.model

    # to_pil = ToPILImage()

    # for batch in valid_loader:
    #     outputs = module(batch[0].to(torch.device("cuda:0")))
    #     outputs[outputs > 0.5] = 1
    #     outputs[outputs <= 0.5] = 0
    #     batch.append(outputs.detach().cpu())

    #     for i in range(batch[0].size(2)):
    #         _, axs = plt.subplots(1, 3)

    #         for j, ax in enumerate(axs.flatten()):
    #             img = to_pil(batch[j][i])
    #             ax.imshow(img)
    #             ax.axis('off')

    #         plt.tight_layout()
    #         plt.show()

    lr_schedulers = [ReduceLROnPlateau(factor=0.5, min_lr=1e-6)]
    exp.train(train_loader=train_loader,
              valid_loader=valid_loader,
              epochs=1000,
              lr_schedulers=lr_schedulers)
