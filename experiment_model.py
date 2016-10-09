import numpy as np
from core.mathhelper import angle_between

class ExperimentModel:
    def get_layers(self):
        pass

    def get_models(self):
        pass

    def get_validation_score(self, nValidateBatches):
        validateModel = self.get_models()[1]

        # Run the validation set.
        validationLosses = [
            validateModel(i)
            for i in range(nValidateBatches)
        ]

        validationScore = np.mean(validationLosses)

        return validationScore

    def get_anchor_vectors(self):
        anchorVecs = []
        for layer in get_layers():
            W = layer.W.get_value(borrow=True)
            # Flatten if necessary.
            if len(W.shape) > 3:
                anchorVecs.append([filterWeight.flatten() for filterWeight in W])
            else:
                anchorVecs.append(W)

        return anchorVecs

    def get_deep_copy_params(self):
        layers = self.get_layers()
        layersParams = []
        for layer in layers:
            W = copy.deepcopy(preTrainedLayer.W.get_value(borrow=False))
            b = copy.deepcopy(preTrainedLayer.b.get_value(borrow=False))
            layersParams.append((W, b))

        return layersParams


def get_anchor_angles(anchors0, anchors1):
    anchors = zip(anchors0, anchors1)

    angles = []
    for anchor in anchors:
        angle = angle_between(anchors[0], anchors[1])
        angles.append(angle)

    return angles
