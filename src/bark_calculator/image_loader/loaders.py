import os
from skimage.io import imread
from pathlib import Path
import glob


class Loader:
    supported_file_types = [".bmp"]
    default_source_path = "Images/"

    def __init__(self, source_path=default_source_path):
        self.source_path = source_path
        self.images_list = []

    def __iter__(self):
        yield from self.images_list

    def build_images_list_from_names(self, image_names):
        for image_name in image_names:
            assert image_name[-4:] in self.supported_file_types, \
                "Unsupported image file type {}".format(image_name[-4:])

            image_path = os.path.join(self.source_path, image_name)
            self.images_list.append([imread(image_path), image_name])


class GoodExamplesLoader(Loader):
    good_examples_path = "res/good_examples.txt"

    def __init__(self, good_examples_path=good_examples_path):
        super().__init__()
        self.good_examples_path = good_examples_path
        self.target_images_names = [image_name.split("\n")[0] for image_name in
                                    open(self.good_examples_path, "r").readlines()]
        self.build_images_list_from_names(self.target_images_names)


class FolderLoader(Loader):

    def __init__(self, folder_path):
        super().__init__()
        self.wood_types = ["epn_gele", "epn_non-gele",
                           "sap_gele"]

        image_paths = []
        for wood_type in self.wood_types:
            wood_type_path = Path(os.path.join(folder_path, wood_type))
            image_paths.extend([(img_path, wood_type) for img_path
                                in wood_type_path.rglob("*.bmp")])

        self.images_list = [[imread(str(p)), t, p.name]
                            for p, t in image_paths]
