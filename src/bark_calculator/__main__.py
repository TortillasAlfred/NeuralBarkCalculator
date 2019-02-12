import argparse
from image_loader.builder import build_from_image_src_arg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("image_src", help="Path to images folder or 'good_examples' for a pre-filtered images source",
                        type=str)

    args = parser.parse_args()

    image_loader = build_from_image_src_arg(args.image_src)
    print("allo")
