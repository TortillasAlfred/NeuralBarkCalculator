from dataset import RegressionDatasetFolder, pil_loader
from utils import *
from models import fcn_resnet50, deeplabv3_resnet50, fcn_resnet101, deeplabv3_resnet101, fcn_efficientnet, deeplabv3_efficientnet
from lovasz_losses import LovaszSoftmax

from torchvision.transforms import *

from poutyne.framework import Experiment, ExponentialLR, EarlyStopping, ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, BatchSampler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.nn.modules.loss import CrossEntropyLoss
from skimage.io import imread, imsave

from sklearn.metrics import f1_score
from skimage.transform import resize
from skimage import img_as_ubyte
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
    levels = [('combined_images', ['train', 'valid', 'test']),
              ('outputs', ['train', 'valid', 'test'])]

    results_dir = os.path.join(root_dir, 'Images', 'results', 'deeplab')

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


def make_dual_images():
    barks_dir = "./Images/sapin/bark"
    nodes_dir = "./Images/sapin/nodes"
    duals_dir = "./Images/sapin/duals"

    for _, _, fnames in sorted(os.walk(barks_dir)):
        for fname in sorted(fnames):
            bark_path = os.path.join(barks_dir, fname)
            node_path = os.path.join(nodes_dir, fname)

            bark_image = np.asarray(pil_loader(bark_path,
                                               grayscale=True)) / 255
            node_image = np.asarray(pil_loader(node_path,
                                               grayscale=True)) / 255

            dual_png = np.zeros((bark_image.shape[0], bark_image.shape[1]),
                                dtype=np.uint8)
            dual_png[bark_image == 1.0] = 127
            dual_png[node_image == 1.0] = 255

            dual = Image.fromarray(dual_png, mode='L')
            dual.save(os.path.join(duals_dir, fname.replace("bmp", "png")))


def fine_tune_images():
    duals_dir = "/home/magod/Documents/Encorcage/Images/1024_with_jedi/duals/"
    output_dir = "/home/magod/Documents/Encorcage/Images/1024_with_jedi_tuned/duals/"

    for wood_type in ["epinette_gelee", "epinette_non_gelee", "sapin"]:
        type_duals_dir = os.path.join(duals_dir, wood_type)
        type_output_dir = os.path.join(output_dir, wood_type)

        for _, _, fnames in sorted(os.walk(type_duals_dir)):
            for fname in sorted(fnames):
                print(fname)

                dual_path = os.path.join(type_duals_dir, fname)

                dual_image = np.array(
                    Image.open(open(dual_path, 'rb')).convert('L')) / 127

                dual_image = remove_small_zones(
                    torch.from_numpy(dual_image).long())

                dual_image = dual_image.numpy().astype(np.uint8)
                dual_image[dual_image == 1] = 127
                dual_image[dual_image == 2] = 255

                dual = Image.fromarray(dual_image, mode='L')
                out_path = os.path.join(type_output_dir, fname)
                dual.save(out_path)


def adjust_images(duals_folder, samples_folder, out_folder):
    for _, _, fnames in sorted(os.walk(duals_folder)):
        for fname in sorted(fnames):
            sample = imread(
                os.path.join(samples_folder, fname.replace(".png", ".bmp")))
            dual = imread(os.path.join(duals_folder, fname), grayscale=True)

            dual = img_as_ubyte(resize(dual, sample.shape[:-1], order=0))

            try:
                dual = Image.fromarray(dual, mode='L')
                dual.save(os.path.join(out_folder, fname))
            except ValueError:
                print(fname)


def test_color_jitter(root_dir):
    train_dataset = RegressionDatasetFolder(
        os.path.join(root_dir, "Images/1024_with_jedi"),
        input_only_transform=Compose([
            ToPILImage(),
            ColorJitter(brightness=0.15, saturation=0.15, contrast=0.15),
            ToTensor()
        ]),
        transform=Compose([
            Lambda(lambda img: pad_resize(img, 1024, 1024)),
            RandomCrop(256),
            ToTensor()
        ]),
        in_memory=True)

    loader = DataLoader(train_dataset,
                        batch_size=1,
                        num_workers=0,
                        pin_memory=False)

    for imgs in loader:
        input = imgs[0][0]
        input = ToPILImage()(input)
        plt.imshow(input)
        plt.show()


def get_loader_for_crop_batch(crop_size, batch_size, train_split, mean, std,
                              train_weights, root_dir):
    train_dataset = RegressionDatasetFolder(
        os.path.join(root_dir, "Images/1024_with_jedi"),
        input_only_transform=Compose([Normalize(mean, std)]),
        transform=Compose([
            Lambda(lambda img: pad_resize(img, 1024, 1024)),
            ColorJitter(brightness=0.15, saturation=0.15, contrast=0.15),
            RandomCrop(crop_size),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ToTensor()
        ]),
        in_memory=True)

    sampler = BatchSampler(WeightedRandomSampler(
        train_weights, num_samples=len(train_weights), replacement=True),
                           batch_size=batch_size)

    return DataLoader(Subset(train_dataset, train_split),
                      batch_sampler=sampler,
                      num_workers=8,
                      pin_memory=False)


def test_model_on_checkpoint(model, test_loader):
    best_epoch_stats = None

    test_loss, test_metrics = model.evaluate_generator(test_loader, steps=None)
    if not isinstance(test_metrics, np.ndarray):
        test_metrics = np.array([test_metrics])

    test_metrics_names = ['test_loss'] + \
                            ['test_' + metric_name for metric_name in model.metrics_names]
    test_metrics_values = np.concatenate(([test_loss], test_metrics))

    test_metrics_dict = {
        col: val
        for col, val in zip(test_metrics_names, test_metrics_values)
    }
    test_metrics_str = ', '.join('%s: %g' % (col, val)
                                 for col, val in test_metrics_dict.items())
    print("On best model: %s" % test_metrics_str)


def main(args):
    raw_dataset = RegressionDatasetFolder(os.path.join(
        args.root_dir, 'Images/1024_with_jedi'),
                                          input_only_transform=None,
                                          transform=Compose([ToTensor()]))
    mean, std = compute_mean_std(raw_dataset)
    print(mean)
    print(std)
    pos_weights = compute_pos_weight(raw_dataset)
    print(pos_weights)
    test_dataset = RegressionDatasetFolder(
        os.path.join(args.root_dir, 'Images/1024_with_jedi'),
        input_only_transform=Compose([Normalize(mean, std)]),
        transform=Compose(
            [Lambda(lambda img: pad_resize(img, 1024, 1024)),
             ToTensor()]),
        in_memory=True)

    valid_dataset = RegressionDatasetFolder(
        os.path.join(args.root_dir, 'Images/1024_with_jedi'),
        input_only_transform=Compose([Normalize(mean, std)]),
        transform=Compose([ToTensor()]),
        include_fname=True)

    train_split, valid_split, test_split, train_weights = get_splits(
        valid_dataset)
    valid_loader = DataLoader(Subset(test_dataset, valid_split),
                              batch_size=8,
                              num_workers=8,
                              pin_memory=False)

    # module = deeplabv3_efficientnet(n=5)
    # module = fcn_resnet50(dropout=0.8)
    module = deeplabv3_resnet50()

    optim = torch.optim.Adam(module.parameters(), lr=1e-3, weight_decay=1e-6)
    exp = Experiment(directory=os.path.join(args.root_dir, 'deeplab'),
                     module=module,
                     device=torch.device(args.device),
                     optimizer=optim,
                     loss_function=LovaszSoftmax(),
                     batch_metrics=[IOU(None)],
                     monitor_metric='val_IntersectionOverUnion',
                     monitor_mode='max')

    lr_schedulers = [ExponentialLR(0.97)]
    callbacks = [
        EarlyStopping(monitor='val_IntersectionOverUnion',
                      min_delta=1e-3,
                      patience=35,
                      verbose=True,
                      mode='max')
    ]

    for i, (crop_size, batch_size) in enumerate(zip([256], [16])):
        train_loader = get_loader_for_crop_batch(crop_size, batch_size,
                                                 train_split, mean, std,
                                                 train_weights, args.root_dir)

        exp.train(train_loader=train_loader,
                  valid_loader=valid_loader,
                  epochs=(1 + i) * 300,
                  lr_schedulers=lr_schedulers,
                  callbacks=callbacks)

    raw_dataset.print_filenames()

    pure_dataset = RegressionDatasetFolder(os.path.join(
        args.root_dir, 'Images/1024_with_jedi'),
                                           transform=Compose([ToTensor()]),
                                           include_fname=True)

    test_loader = DataLoader(Subset(test_dataset, test_split),
                             batch_size=8,
                             num_workers=8,
                             pin_memory=False)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=1,
                              num_workers=8,
                              pin_memory=False)
    pure_loader = DataLoader(pure_dataset,
                             batch_size=1,
                             num_workers=8,
                             pin_memory=False)

    exp.test(test_loader)

    exp.load_best_checkpoint()
    module = exp.model.model
    module.eval()

    generate_output_folders(args.root_dir)

    splits = [(train_split, 'train'), (valid_split, 'valid'),
              (test_split, 'test')]

    results_csv = [[
        'Name', 'Type', 'Split', 'F1_nothing', 'F1_bark', 'F1_node', 'F1_mean',
        'Output Bark %', 'Output Node %', 'Target Bark %', 'Target Node %'
    ]]

    with torch.no_grad():
        for image_number, (batch, pure_batch) in enumerate(
                zip(valid_loader, pure_loader)):
            input = pure_batch[0]
            target = pure_batch[1]
            fname = pure_batch[2][0]
            wood_type = pure_batch[3][0]

            del pure_batch

            outputs = module(batch[0].to(torch.device(args.device)))
            outputs = remove_small_zones(outputs)

            del batch

            names = ['Input', 'Target', 'Generated image']

            try:
                class_accs = IOU(class_to_watch='all')(outputs, target)

                acc = class_accs.mean()
            except ValueError as e:
                print('Error on file {}'.format(fname))
                print(outputs.shape)
                print(target.shape)
                raise e

            outputs = torch.argmax(outputs, dim=1)

            imgs = [input, target, outputs]
            imgs = [img.detach().cpu().squeeze().numpy() for img in imgs]

            fig, axs = plt.subplots(1, 3)

            class_names = ['Nothing', 'Bark', 'Node']

            for i, ax in enumerate(axs.flatten()):
                img = imgs[i]

                raw = (len(img.shape) == 3)

                if raw:  # Raw input
                    img = img.transpose(1, 2, 0)

                values = np.unique(img.ravel())

                plotted_img = ax.imshow(img, vmax=2)
                ax.set_title(names[i])
                ax.axis('off')

                if not raw:  # Predicted image
                    patches = [
                        mpatches.Patch(
                            color=plotted_img.cmap(plotted_img.norm(value)),
                            label='{} zone'.format(class_names[value]))
                        for value in values
                    ]

            suptitle = 'Mean f1 : {:.3f}'.format(acc)

            for split_idxs, split_name in splits:
                if image_number in split_idxs:
                    split = split_name

            running_csv_stats = [fname, wood_type, split]

            class_names = ['Nothing', 'Bark', 'Node']

            for c, c_acc in zip(class_names, class_accs):
                suptitle += '\n{} : {:.3f}'.format(c, c_acc)
                running_csv_stats.append('{:.3f}'.format(c_acc))

            running_csv_stats.append('{:.3f}'.format(acc))

            for class_idx in [1, 2]:
                class_percent = (outputs == class_idx).float().mean().cpu()
                running_csv_stats.append('{:.5f}'.format(class_percent * 100))

            for class_idx in [1, 2]:
                class_percent = (target == class_idx).float().mean().cpu()
                running_csv_stats.append('{:.5f}'.format(class_percent * 100))

            fig.legend(handles=patches,
                       title='Classes',
                       bbox_to_anchor=(0.4, -0.2, 0.5, 0.5))
            plt.suptitle(suptitle)
            plt.tight_layout()
            # plt.show()
            plt.savefig(os.path.join(
                args.root_dir,
                'Images/results/deeplab/combined_images/{}/{}/{}').format(
                    wood_type, split, fname),
                        format='png',
                        dpi=900)
            plt.close()

            outputs = outputs.squeeze().cpu().numpy()
            dual_outputs = np.zeros((outputs.shape[0], outputs.shape[1]),
                                    dtype=np.uint8)
            dual_outputs[outputs == 1] = 127
            dual_outputs[outputs == 2] = 255

            dual = Image.fromarray(dual_outputs, mode='L')
            dual.save(
                os.path.join(args.root_dir,
                             'Images/results/deeplab/outputs/{}/{}/{}').format(
                                 wood_type, split, fname))

            results_csv.append(running_csv_stats)

    csv_file = os.path.join(args.root_dir, 'Images', 'results', 'deeplab',
                            'final_stats.csv')

    with open(csv_file, 'w') as f:
        csv_writer = csv.writer(f, delimiter='\t')
        csv_writer.writerows(results_csv)


def fix_image(img_number, n_pixels_to_fix, which_to_reduce):
    dual = imread(
        "/home/magod/Documents/Encorcage/Images/1024_with_jedi/duals/epinette_gelee/{}.png"
        .format(img_number))
    sample = imread(
        "/home/magod/Documents/Encorcage/Images/1024_with_jedi/samples/epinette_gelee/{}.bmp"
        .format(img_number))

    if which_to_reduce == 'sample':
        img = sample
        output_path = "/home/magod/Documents/Encorcage/Images/1024_with_jedi/samples/epinette_gelee/{}.bmp".format(
            img_number)
    else:
        img = dual
        output_path = "/home/magod/Documents/Encorcage/Images/1024_with_jedi/duals/epinette_gelee/{}.png".format(
            img_number)

    if n_pixels_to_fix == 1:
        img = img[:-1]
    elif n_pixels_to_fix == 2:
        img = img[1:-1]
    else:
        raise ValueError()

    imsave(output_path, img)


if __name__ == "__main__":
    # fix_image('EPN 9 A', 1, "smple")
    # make_dual_images()
    # fine_tune_images()
    # adjust_images("./Images/epinette_gelee_retrieved", "./Images/1024_processed/samples/epinette_gelee/",
    #               "./Images/adjusted/")

    parser = argparse.ArgumentParser()

    parser.add_argument('root_dir', type=str, help='root directory path.')

    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='Which torch device to train with.',
                        choices=['cpu', 'cuda:0', 'cuda:1'])

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='Which random seed to use.')

    args = parser.parse_args()

    make_training_deterministic(args.seed)

    # test_color_jitter(args.root_dir)
    main(args)
