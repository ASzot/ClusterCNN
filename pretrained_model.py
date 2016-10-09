from experiment_model import ExperimentModel
from experiment import load_model
from experiment import save_model
from model_builder import build_lenet_model
from experiment import create_pretrained

class PretrainedModel(ExperimentModel):
    def __init__(self, forceCreate, rng, batchSize, batchIndex, x, y, nkerns, filterShape, stride, inputShape, datasets, learningRate, saveLocation='models/pretrained.h5'):
        # if not forceCreate:
        #     preTrainedLayers = load_model(saveLocation, rng, batchSize, x, inputShape, nkerns, filterShape)
        # else:
        #     save_model(preTrainedLayers, saveLocation)
        preTrainedLayers = create_pretrained(datasets, filterShape, stride, inputShape, nkerns, rng, batchSize, x, y, batchIndex, forceCreate)

        models = build_lenet_model(datasets, preTrainedLayers, x, y, batchIndex, batchSize, learningRate)

        self.layers = preTrainedLayers
        self.models = models

    def get_layers(self):
        return self.layers

    def get_models(self):
        return self.models
