from dataset import RegressionDatasetFolder, make_weight_map, pil_loader
from utils import *
from models import vanilla_unet, FCDenseNet103, FCDenseNet57, B2B, deeplabv3_resnet101

from torchvision.transforms import *

from poutyne.framework import Experiment, ReduceLROnPlateau, EarlyStopping
from torch.utils.data import DataLoader, Subset, ConcatDataset
import matplotlib.pyplot as plt
from torch.nn.modules.loss import CrossEntropyLoss
from skimage.io import imread, imsave

from sklearn.metrics import f1_score
import torch

from math import ceil
import numpy as np
import io
import pickle
from PIL import Image
import os


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


def main():
    # mean, std = compute_mean_std("./Images/dual_exp")
    # pos_weights = compute_pos_weight("./Images/dual_exp")
    mean, std = get_mean_std()
    pos_weights = get_pos_weight()
    test_dataset = RegressionDatasetFolder("/mnt/storage/mgodbout/Ecorcage/Images/dual_exp",
                                           input_only_transform=Compose(
                                               [Normalize(mean, std)]
                                           ),
                                           transform=Compose([
                                               ToTensor()]))

    train_dataset = RegressionDatasetFolder("/mnt/storage/mgodbout/Ecorcage/Images/dual_exp",
                                            input_only_transform=Compose(
                                                [Normalize(mean, std)]
                                            ),
                                            transform=Compose([
                                                RandomCrop(448),
                                                RandomHorizontalFlip(),
                                                RandomVerticalFlip(),
                                                ToTensor()]))

    train_split, valid_split, test_split = get_splits(train_dataset)

    train_loader = DataLoader(Subset(train_dataset, train_split.repeat(25)), batch_size=6, shuffle=True)
    valid_loader = DataLoader(Subset(test_dataset, np.hstack((valid_split, train_split))), batch_size=1)
    test_loader = DataLoader(Subset(test_dataset, test_split), batch_size=1)

    module = deeplabv3_resnet101()

    optim = torch.optim.Adam(module.parameters(), lr=1e-3)
    exp = Experiment(directory="/mnt/storage/mgodbout/Ecorcage/cwce_448/",
                     module=module,
                     device=torch.device("cuda:1"),
                     optimizer=optim,
                     loss_function=CustomWeightedCrossEntropy(torch.tensor(pos_weights).to('cuda:1')),
                     metrics=[IOU(None), IOU(0), IOU(1), IOU(2)],
                     monitor_metric='val_IntersectionOverUnion',
                     monitor_mode='max')

    lr_schedulers = [ReduceLROnPlateau(patience=10, monitor='val_IntersectionOverUnion', mode='max')]
    callbacks = [EarlyStopping(patience=30, min_delta=1e-5)]
    exp.train(train_loader=train_loader,
              valid_loader=valid_loader,
              epochs=1500,
              lr_schedulers=lr_schedulers,
              callbacks=callbacks)
    exp.test(valid_loader)
    exp.test(test_loader)

    module = exp.model.model
    module.eval()

    valid_dataset = RegressionDatasetFolder("/mnt/storage/mgodbout/Ecorcage/Images/dual_exp",
                                            input_only_transform=Compose(
                                                [Normalize(mean, std)]
                                            ),
                                            transform=Compose([
                                                ToTensor()]),
                                            include_fname=True)
    pure_dataset = RegressionDatasetFolder("/mnt/storage/mgodbout/Ecorcage/Images/dual_exp",
                                           transform=Compose([
                                               ToTensor()]),
                                           include_fname=True)

    valid_loader = DataLoader(valid_dataset, batch_size=1)
    pure_loader = DataLoader(pure_dataset, batch_size=1)

    if not os.path.isdir("/mnt/storage/mgodbout/Ecorcage/Images/results/cwce_448"):
        os.makedirs("/mnt/storage/mgodbout/Ecorcage/Images/results/cwce_448")

    with torch.no_grad():
        for batch, pure_batch in zip(valid_loader, pure_loader):
            input = pure_batch[0]
            target = pure_batch[1]
            fname = pure_batch[2][0]

            del pure_batch

            # if os.path.isfile("/mnt/storage/mgodbout/Ecorcage/Images/results/cwce_448/{}".format(fname)):
            #     continue

            outputs = module(batch[0].to(torch.device("cuda:1")))
            outputs = torch.argmax(outputs, dim=1)

            del batch

            names = ["Input", "Target", "Generated image"]

            imgs = [input, target, outputs]
            imgs = [img.detach().cpu().squeeze().numpy() for img in imgs]

            try:
                class_accs = f1_score(imgs[1].flatten(), imgs[2].flatten(), labels=[0, 1, 2], average=None)
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

            plt.suptitle(suptitle)
            plt.tight_layout()
            # plt.show()
            plt.savefig("/mnt/storage/mgodbout/Ecorcage/Images/results/cwce_448/{}".format(fname),
                        format="png",
                        dpi=900)


def fix_image(img_number, n_pixels_to_fix):
    dual = imread("/home/magod/Documents/Encorcage/Images/dual_exp/duals/{}.png".format(img_number))
    bark = imread("/home/magod/Documents/Encorcage/Images/dual_exp/bark/{}.bmp".format(img_number))
    node = imread("/home/magod/Documents/Encorcage/Images/dual_exp/nodes/{}.bmp".format(img_number))
    sample = imread("/home/magod/Documents/Encorcage/Images/dual_exp/samples/{}.bmp".format(img_number))

    if n_pixels_to_fix == 1:
        dual = dual[:-1]
        bark = bark[:-1]
        node = node[:-1]
    elif n_pixels_to_fix == 2:
        dual = dual[1:-1]
        bark = bark[1:-1]
        node = node[1:-1]
    else:
        raise ValueError()

    imsave("/home/magod/Documents/Encorcage/Images/dual_exp/duals/{}.png".format(img_number), dual)
    imsave("/home/magod/Documents/Encorcage/Images/dual_exp/bark/{}.bmp".format(img_number), bark)
    imsave("/home/magod/Documents/Encorcage/Images/dual_exp/nodes/{}.bmp".format(img_number), node)


if __name__ == "__main__":
    # fix_image(264, 1)
    # make_dual_images()
    main()
