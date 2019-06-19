import argparse

from experiments.builder import build_experiment_from_args


def parse_all_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("image_loader", help="How to load images to process. Can be an image folder " + \
                        "or 'good_examples' for a pre-filtered images source", type=str)
    parser.add_argument("--treatment_method", help="Treatment method used when calculating an image's bark. " + \
                        "Possible values are : 'edge_detection', 'id'", type=str, default="edge_detection")
    parser.add_argument("--image_processor", help="Process applied to every image once treated. " + \
                        "Possible values are : 'display'", type=str, default="display")

    return parser.parse_args()


if __name__ == "__main__":
    exp = build_experiment_from_args(parse_all_args())
    exp.run()
