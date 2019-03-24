

class GenericExperiment:

    def __init__(self, image_loader, treatment_method, processor):
        self.image_loader = image_loader
        self.treatment_method = treatment_method
        self.processor = processor

    def run(self):
        self.processor.process(self.image_loader, self.treatment_method)