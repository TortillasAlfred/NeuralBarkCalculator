from models import NeuralBarkCalculator, Preprocessor

import os
import argparse
import time
import multiprocessing
import torch


def generate_folders(root_path, only_preprocess):
    def mkdirs_if_not_there(dir):
        if not os.path.isdir(dir):
            os.makedirs(dir)

    all_wood_types = ['epinette_gelee', 'epinette_non_gelee', 'sapin']
    present_wood_types = os.listdir(os.path.join(root_path, 'samples'))
    wood_types = list(set(all_wood_types) & set(present_wood_types))

    # Processed folders
    levels = ['samples']

    processed_dir = os.path.join(root_path, 'processed')

    for folder in levels:
        current_dir = os.path.join(processed_dir, folder)

        mkdirs_if_not_there(current_dir)

        for wood_type in wood_types:
            wood_dir = os.path.join(current_dir, wood_type)

            mkdirs_if_not_there(wood_dir)

    if not only_preprocess:
        # Results folders
        levels = ['combined_images', 'outputs']

        results_dir = os.path.join(root_path, 'results')

        for folder in levels:
            current_dir = os.path.join(results_dir, folder)

            mkdirs_if_not_there(current_dir)

            for wood_type in wood_types:
                wood_dir = os.path.join(current_dir, wood_type)

                mkdirs_if_not_there(wood_dir)


def main(args):
    generate_folders(args.root_path, args.only_preprocess)

    Preprocessor().preprocess_images(args.root_path)

    if not args.only_preprocess:
        model = NeuralBarkCalculator('./best_model.pt', args.device)
        model.predict(args.root_path, args.exclude_nodes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('root_path', type=str, help='root directory path.')

    parser.add_argument('--device',
                        type=str,
                        default='cpu',
                        help='Which torch device to train with.',
                        choices=['cpu', 'cuda:0', 'cuda:1'])

    parser.add_argument('--exclude_nodes', action='store_true', default=False)

    parser.add_argument('--only_preprocess', action='store_true', default=False)

    args = parser.parse_args()

    if args.device == 'cpu':
        torch.set_num_threads(multiprocessing.cpu_count())

    main(args)
