class ModelWrapper(object):
    def __init__(self, accuracy, centroids, model):
        self.accuracy = accuracy
        self.centroids = centroids
        self.model = model
