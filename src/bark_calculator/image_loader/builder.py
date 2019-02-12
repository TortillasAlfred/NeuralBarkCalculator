from image_loader.loaders import *


def build_from_image_src_arg(image_src):
    if image_src == "good_examples":
        return GoodExamplesLoader()
