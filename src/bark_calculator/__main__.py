from dataset import RegressionDatasetFolder, make_weight_map, pil_loader
from utils import *
from models import vanilla_unet, FCDenseNet103, FCDenseNet57, B2B, deeplabv3_resnet101

from torchvision.transforms import *

from poutyne.framework import Experiment, ReduceLROnPlateau, EarlyStopping
from torch.utils.data import DataLoader, Subset, ConcatDataset
import matplotlib.pyplot as plt
from torch.nn.modules.loss import CrossEntropyLoss
import torch

from math import ceil
import numpy as np
import io
import pickle
from PIL import Image
import os


def load_best_and_show(exp, pure_loader, valid_loader):
    exp.load_best_checkpoint()
    module = exp.model.model
    module.to(torch.device("cuda:0"))
    module.eval()

    to_pil = ToPILImage()

    for batch, pure_batch in zip(valid_loader, pure_loader):
        outputs = module(batch[0].to(torch.device("cuda:0")))
        torch.sigmoid(outputs)
        outputs[outputs > 0.5] = 1
        outputs[outputs <= 0.5] = 0
        batch.append(outputs.detach().cpu())
        batch[0] = pure_batch[0]
        tmp = batch[2]
        batch[2] = batch[3]
        batch[3] = tmp

        names = ["Input", "Target", "Generated image"]

        for i in range(batch[1].size(0)):
            _, axs = plt.subplots(1, 3)
            acc = (batch[2][i] == batch[1][i]).sum().item()/(256 * 256)

            for j, ax in enumerate(axs.flatten()):
                img = to_pil(batch[j][i])
                ax.imshow(img)
                ax.set_title(names[j])
                ax.axis('off')

            plt.suptitle("Overall accuracy : {:.3f}".format(acc))
            plt.tight_layout()
            plt.show()
            # plt.savefig("Images/results/nn_cut_mix/{}".format(batch[3][i]),
            #             format="png",
            #             dpi=900)


def show_dataset(dataset, pure_dataset, augmented_dataset):
    for sample, augmented_sample in zip(iter(pure_dataset), iter(augmented_dataset)):
        _,  axs = plt.subplots(2, 2)

        sample += augmented_sample

        for ax, img in zip(axs.flatten(), sample):
            img = ToPILImage()(img)
            ax.imshow(img)
            ax.axis('off')

        plt.tight_layout()
        plt.show()


def old_main():
    mean, std = get_mean_std()
    dataset = RegressionDatasetFolder("/mnt/storage/mgodbout/Ecorcage/Images/nn_cut",
                                      input_only_transform=Compose(
                                          [Normalize(mean, std)]
                                      ),
                                      transform=Compose(
                                          [Lambda(lambda img:
                                                  pad_resize(img, 1024, 1024)),
                                           ToTensor()]
                                      ),
                                      mode='all')
    pure_dataset = RegressionDatasetFolder("/mnt/storage/mgodbout/Ecorcage/Images/nn_cut",
                                           transform=Compose(
                                               [Resize((1024, 1024)),
                                                ToTensor()]
                                           ),
                                           mode='all')
    augmented_dataset = RegressionDatasetFolder("/mnt/storage/mgodbout/Ecorcage/Images/nn_cut",
                                                input_only_transform=Compose(
                                                    [Normalize(mean, std)]
                                                ),
                                                transform=Compose([
                                                    RandomHorizontalFlip(),
                                                    RandomVerticalFlip(),
                                                    Lambda(lambda img:
                                                           pad_resize(img, 1024, 1024)),
                                                    RandomResizedCrop(
                                                        1024, scale=(0.8, 1.0)),
                                                    ToTensor()]),
                                                mode='all')

    # show_dataset(dataset, pure_dataset, augmented_dataset)

    train_sampler, valid_sampler = get_train_valid_samplers(dataset,
                                                            train_percent=0.8)
    train_loader = DataLoader(augmented_dataset, batch_size=8,
                              sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=8,
                              sampler=valid_sampler)
    pure_loader = DataLoader(pure_dataset, batch_size=8,
                             sampler=valid_sampler)
    module = vanilla_unet()
    optim = torch.optim.Adam(module.parameters(), lr=1e-3, weight_decay=1e-5)
    exp = Experiment(directory="/mnt/storage/mgodbout/Ecorcage/weighted_unet/",
                     module=module,
                     device=torch.device("cuda:0"),
                     optimizer=optim,
                     loss_function=MixedLoss())

    # load_best_and_show(exp, pure_loader, valid_loader)

    lr_schedulers = [ExponentialLR(gamma=0.98)]
    exp.train(train_loader=train_loader,
              valid_loader=valid_loader,
              epochs=500,
              lr_schedulers=lr_schedulers)


def make_dual_images():
    barks_dir = "/mnt/storage/mgodbout/Ecorcage/Images/dual_exp/bark"
    nodes_dir = "/mnt/storage/mgodbout/Ecorcage/Images/dual_exp/nodes"
    duals_dir = "/mnt/storage/mgodbout/Ecorcage/Images/dual_exp/duals"

    for _, _, fnames in sorted(os.walk(barks_dir)):
        for fname in sorted(fnames):
            bark_path = os.path.join(barks_dir, fname)
            node_path = os.path.join(nodes_dir, fname)

            bark_image = np.asarray(pil_loader(bark_path, grayscale=True))/255
            node_image = np.asarray(pil_loader(node_path, grayscale=True))/255

            dual_png = np.zeros((bark_image.shape[0], bark_image.shape[1]), dtype=np.uint8)
            dual_png[bark_image == 1.0] = 127
            dual_png[node_image == 1.0] = 255

            dual = Image.fromarray(dual_png, mode='L')
            dual.save(os.path.join(duals_dir, fname.replace("bmp", "png")))


def new_main():
    # make_dual_images()
    mean, std = get_mean_std()
    pos_weights = get_pos_weights()
    test_dataset = RegressionDatasetFolder("/mnt/storage/mgodbout/Ecorcage/Images/nn_cut",
                                           input_only_transform=Compose(
                                               [Normalize(mean, std)]
                                           ),
                                           transform=Compose([
                                               Lambda(lambda img:
                                                      pad_resize(img, 1024, 1024)),
                                               ToTensor()]))

    for k in range(1, 6):
        train_dataset = RegressionDatasetFolder("/mnt/storage/mgodbout/Ecorcage/Images/nn_cut",
                                                input_only_transform=Compose(
                                                    [Normalize(mean, std)]
                                                ),
                                                transform=Compose([
                                                    RandomHorizontalFlip(),
                                                    RandomVerticalFlip(),
                                                    Lambda(lambda img:
                                                           pad_resize(img, 1024, 1024)),
                                                    ToTensor()]),
                                                k=k,
                                                mode="train")
        valid_dataset = RegressionDatasetFolder("/mnt/storage/mgodbout/Ecorcage/Images/nn_cut",
                                                input_only_transform=Compose(
                                                    [Normalize(mean, std)]
                                                ),
                                                transform=Compose([
                                                    Lambda(lambda img:
                                                           pad_resize(img, 1024, 1024)),
                                                    ToTensor()]),
                                                k=k,
                                                mode="valid")

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=8)
        test_loader = DataLoader(test_dataset, batch_size=8)

        module = vanilla_unet()
        optim = torch.optim.Adam(
            module.parameters(), lr=1e-2, weight_decay=1e-5)
        exp = Experiment(directory="/mnt/storage/mgodbout/Ecorcage/weighted_unet/{}/".format(k),
                         module=module,
                         device=torch.device("cuda:0"),
                         optimizer=optim,
                         loss_function=BCEWithLogitsLoss(weight=pos_weights))

        lr_schedulers = [ExponentialLR(gamma=0.98)]
        callbacks = [EarlyStopping(patience=20, min_delta=1e-5)]
        exp.train(train_loader=train_loader,
                  valid_loader=valid_loader,
                  epochs=500,
                  lr_schedulers=lr_schedulers,
                  callbacks=callbacks)
        exp.test(test_loader)

    test_loader = DataLoader(test_dataset, batch_size=1)
    module = B2B("/mnt/storage/mgodbout/Ecorcage/weighted_unet/", 5)
    exp = Experiment(directory="/mnt/storage/mgodbout/Ecorcage/weighted_unet/",
                     module=module,
                     device=torch.device("cuda:0"),
                     loss_function=MixedLoss())
    exp.test(test_loader, load_best_checkpoint=False)

    with open("/mnt/storage/mgodbout/Ecorcage/weighted_unet/ensemble.pck", "wb") as f:
        pickle.dump(exp.model.model, f,
                    pickle.HIGHEST_PROTOCOL)

    module = pickle.load(
        open("/mnt/storage/mgodbout/Ecorcage/weighted_unet/ensemble.pck",
             "rb"))

    module.to(torch.device("cuda:0"))
    module.eval()

    to_pil = ToPILImage()

    valid_dataset = RegressionDatasetFolder("/mnt/storage/mgodbout/Ecorcage/Images/nn_cut",
                                            input_only_transform=Compose(
                                                [Normalize(mean, std)]
                                            ),
                                            transform=Compose([
                                                Lambda(lambda img:
                                                       pad_resize(img, 1024, 1024)),
                                                ToTensor()]),
                                            mode="all",
                                            include_fname=True)
    pure_dataset = RegressionDatasetFolder("/mnt/storage/mgodbout/Ecorcage/Images/nn_cut",
                                           transform=Compose([
                                               Lambda(lambda img:
                                                      pad_resize(img, 1024, 1024)),
                                               ToTensor()]),
                                           mode="all",
                                           include_fname=True)

    valid_loader = DataLoader(valid_dataset, batch_size=1)
    pure_loader = DataLoader(pure_dataset, batch_size=1)

    for batch, pure_batch in zip(valid_loader, pure_loader):
        outputs = module(batch[0].to(torch.device("cuda:0")))
        outputs = torch.sigmoid(outputs)
        outputs.round_()
        batch.append(outputs.detach().cpu())
        batch[0] = pure_batch[0]
        tmp = batch[2]
        batch[2] = batch[3]
        batch[3] = tmp

        names = ["Input", "Target", "Generated image"]

        for i in range(batch[1].size(0)):
            _, axs = plt.subplots(1, 3)
            acc = (batch[2][i] == batch[1][i]).sum().item()/(1024 * 1024)
            loss = MixedLoss()(batch[2][i], batch[1][i])

            for j, ax in enumerate(axs.flatten()):
                img = to_pil(batch[j][i])
                ax.imshow(img)
                ax.set_title(names[j])
                ax.axis('off')

            plt.suptitle(
                "Overall accuracy : {:.3f}\n Loss : {:.3f}".format(acc, loss))
            plt.tight_layout()
            # plt.show()
            plt.savefig("/mnt/storage/mgodbout/Ecorcage/Images/results/weighted_unet/{}".format(batch[3][i]),
                        format="png",
                        dpi=900)


def new_new_main():
    # make_dual_images()
    mean, std = get_mean_std()
    pos_weights = get_pos_weight()
    test_dataset = RegressionDatasetFolder("/mnt/storage/mgodbout/Ecorcage/Images/dual_exp",
                                           input_only_transform=Compose(
                                               [Normalize(mean, std)]
                                           ),
                                           transform=Compose([
                                               Lambda(lambda img:
                                                      pad_resize(img, 4096, 4096)),
                                               Resize(2048),
                                               RandomCrop(224),
                                               ToTensor()]))

    train_dataset = RegressionDatasetFolder("/mnt/storage/mgodbout/Ecorcage/Images/dual_exp",
                                            input_only_transform=Compose(
                                                [Normalize(mean, std)]
                                            ),
                                            transform=Compose([
                                                Lambda(lambda img:
                                                       pad_resize(img, 4096, 4096)),
                                                Resize(2048),
                                                RandomCrop(224),
                                                RandomHorizontalFlip(),
                                                RandomVerticalFlip(),
                                                ToTensor()]))
    valid_dataset = RegressionDatasetFolder("/mnt/storage/mgodbout/Ecorcage/Images/dual_exp",
                                            input_only_transform=Compose(
                                                [Normalize(mean, std)]
                                            ),
                                            transform=Compose([
                                                Lambda(lambda img:
                                                       pad_resize(img, 4096, 4096)),
                                                Resize(2048),
                                                RandomCrop(224),
                                                ToTensor()]))

    train_split, valid_split, test_split = get_splits(train_dataset)

    train_loader = DataLoader(ConcatDataset([Subset(train_dataset, train_split)] * 10), batch_size=24, shuffle=True)
    valid_loader = DataLoader(ConcatDataset([Subset(valid_dataset, valid_split)] * 10), batch_size=24)
    test_loader = DataLoader(ConcatDataset([Subset(test_dataset, test_split)] * 10), batch_size=24)

    module = deeplabv3_resnet101()

    optim = torch.optim.Adam(
        module.parameters(), lr=1e-3)
    exp = Experiment(directory="/mnt/storage/mgodbout/Ecorcage/deeplab_focal/",
                     module=module,
                     device=torch.device("cuda:0"),
                     optimizer=optim,
                     loss_function=FocalLossWrapper(),
                     metrics=[IOU()])

    lr_schedulers = [ReduceLROnPlateau(patience=4)]
    callbacks = [EarlyStopping(patience=15, min_delta=1e-5)]
    exp.train(train_loader=train_loader,
              valid_loader=valid_loader,
              epochs=1500,
              lr_schedulers=lr_schedulers,
              callbacks=callbacks)
    exp.test(test_loader)

    module = exp.model.model
    module.eval()

    to_pil = ToPILImage()

    valid_dataset = RegressionDatasetFolder("/mnt/storage/mgodbout/Ecorcage/Images/dual_exp",
                                            input_only_transform=Compose(
                                                [Normalize(mean, std)]
                                            ),
                                            transform=Compose([
                                                Lambda(lambda img:
                                                       pad_resize(img, 4096, 4096)),
                                                Resize(2048),
                                                ToTensor()]),
                                            include_fname=True)
    pure_dataset = RegressionDatasetFolder("/mnt/storage/mgodbout/Ecorcage/Images/dual_exp",
                                           transform=Compose([
                                               Lambda(lambda img:
                                                      pad_resize(img, 4096, 4096)),
                                               Resize(2048),
                                               ToTensor()]),
                                           include_fname=True)

    valid_loader = DataLoader(valid_dataset, batch_size=1)
    pure_loader = DataLoader(pure_dataset, batch_size=1)

    with torch.no_grad():
        for batch, pure_batch in zip(valid_loader, pure_loader):
            outputs = module(batch[0].to(torch.device("cuda:0")))
            outputs = torch.sigmoid(outputs)
            outputs = torch.argmax(outputs, dim=1)
            batch.append(outputs.detach().cpu())
            batch[0] = pure_batch[0]
            tmp = batch[2]
            batch[2] = batch[3]
            batch[3] = tmp

            names = ["Input", "Target", "Generated image"]

            for i in range(batch[1].size(0)):
                _, axs = plt.subplots(1, 3)
                acc = (batch[2][i] == batch[1][i]).sum().item()/(2048 * 2048)

                for j, ax in enumerate(axs.flatten()):
                    img = batch[j][i].detach()

                    if len(img.shape) == 3:
                        img = img.permute(1, 2, 0)

                    ax.imshow(img)
                    ax.set_title(names[j])
                    ax.axis('off')

                plt.suptitle(
                    "Overall accuracy : {:.3f}".format(acc))
                plt.tight_layout()
                # plt.show()
                plt.savefig("/mnt/storage/mgodbout/Ecorcage/Images/results/deeplab_focal/{}".format(batch[3][i]),
                            format="png",
                            dpi=900)


if __name__ == "__main__":
    new_new_main()
