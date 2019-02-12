

class TreatmentMethod:

    def __init__(self):
        pass # Nothing to do here

    def treat_image(self, image):
        raise NotImplementedError("Should be implemented by children classes.")
        
    def treat_images(self, images):
        return [self.treat_image(image) for image in images]
        

class EdgeDetection(TreatmentMethod):

    def __init__(self):
        super().__init__()

    def treat_image(self, image):
        pass
