import argparse
from image_loader.builder import build_from_image_loader_arg
from treatment_method.builder import build_from_treatment_method_arg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("image_loader", help="How to load images to process. Can be an image folder " + \
                        "or 'good_examples' for a pre-filtered images source", type=str)
    parser.add_argument("--treatment_method", help="Treatment method used when calculating an image's bark. " + \
                        "Possible values are : 'edge_detection'", type=str, default="edge_detection")                    

    args = parser.parse_args()

    image_loader = build_from_image_loader_arg(args.image_loader)
    treatment_method = build_from_treatment_method_arg(args.treatment_method)
