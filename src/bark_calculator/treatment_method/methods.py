

class TreatmentMethod:

    def treat_image(self, image):
        raise NotImplementedError("Should be implemented by children classes.")
        
    def treat_images(self, images):
        return [self.treat_image(image) for image in images]
        

class EdgeDetection(TreatmentMethod):

    def treat_image(self, image):
        pass


class Identity(TreatmentMethod):

    def treat_image(self, image):
        return image
