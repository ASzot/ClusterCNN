import keras.callbacks.Callback
from helpers.mathhelper import get_anchor_vectors
from helpers.mathhelper import set_anchor_vectors
import numpy as np

class AnchorVecNormalizer(keras.callbacks.Callback):
    def __init__(self, filter_size, nkerns):
        self.filter_size = filter_size
        self.nkerns = nkerns


    def on_batch_end(self, batch, logs={}):
        # Normalize all of the filter weights for the model.
        anchor_vecs = get_anchor_vectors(self.model)

        # Normalize all of the anchor vectors.
        for i in range(len(anchor_vecs)):
            anchor_vecs[i] = anchor_vecs[i] / np.linalg.norm(anchor_vecs[i])

        set_anchor_vectors(model, anchor_vecs, self.nkerns, self.filter_size))
