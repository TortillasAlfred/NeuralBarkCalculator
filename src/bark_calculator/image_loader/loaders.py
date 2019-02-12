

class Loader:
    supported_file_types = [".bmp"]
    default_source_path = "Images/"

    def __init__(self, source_path=default_source_path):
        self.source_path = source_path
        self.images_list = []
        self.images_iter = None

    def __iter__(self):
        return self.images_iter.next()
        
        
class GoodExamplesLoader(Loader):
    good_examples_path = "res/good_examples.txt"

    def __init__(self, good_examples_path=good_examples_path):
        super().__init__()
        self.good_examples_path = good_examples_path
        self.target_images_names = [image_name.split("\n")[0] for image_name in 
                                    open(self.good_examples_path, "r").readlines()]
