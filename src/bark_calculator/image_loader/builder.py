from image_loader.loaders import *


def build_from_image_loader_arg(image_loader):
    if image_loader == "good_examples":
        return GoodExamplesLoader()

    else:
        return FolderLoader(image_loader)
