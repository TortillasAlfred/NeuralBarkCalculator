import argparse
from image_loader.builder import build_from_image_loader_arg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("image_loader", help="How to load images to process. Can be an image folder " + \
                        "or 'good_examples' for a pre-filtered images source", type=str)

    args = parser.parse_args()

    image_loader = build_from_image_loader_arg(args.image_loader)
