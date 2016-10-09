from experiment_model import ExperimentModel

class TrainedModel(ExperimentModel):
    def __init__(self, forceCreate, initLayers, rng, batchSize, batchIndex, inputShape, x, y, nkerns, filterShape,
                datasets, learningRate, nEpochs, saveLocation='models/posttrained.h5'):
        if not forceCreate:
            postTrainedLayers = load_model(saveLocation, rng, batchSize, x, inputShape, nkerns, filterShape)
            models = build_lenet_model(datasets, initLayers, x, y, batchIndex, batchSize, learningRate)
        else:
            models = build_lenet_model(datasets, initLayers, x, y, batchIndex, batchSize, learningRate)
            trainModel = models[0]
            # Now train the algorithm using SGD
            postTrainedLayers = train_lenet5(datasets[0], trainModel, initLayers, rng, batchSize, batchIndex, nEpochs)
            save_model(postTrainedLayers, saveLocation)
