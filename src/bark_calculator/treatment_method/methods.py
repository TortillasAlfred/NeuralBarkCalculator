

class TreatmentMethod:

    def __init__(self):
        pass # Nothing to do here

    def process_image(self, image):
        raise NotImplementedError("Should be implemented by children classes.")
        
    def process_images(self, images):
        return [self.process_image(image) for image in images]
        

class EdgeDetection(TreatmentMethod):

    def __init__(self):
        super().__init__()

    def process_image(self, image):
        pass
