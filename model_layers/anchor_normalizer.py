from keras.callbacks import Callback
from helpers.mathhelper import get_anchor_vectors
from helpers.mathhelper import set_anchor_vectors
import numpy as np

class AnchorVecNormalizer(Callback):
    def __init__(self, filter_size, nkerns):
        self.filter_size = filter_size
        self.nkerns = nkerns


    def on_batch_end(self, batch, logs={}):
        # Normalize all of the filter weights for the model.
        anchor_vecs = get_anchor_vectors(self.model)

        # Only normalize the first and second convolution layer.
        for i in range(2):
            anchor_vecs[i] = anchor_vecs[i] / np.linalg.norm(anchor_vecs[i])

        set_anchor_vectors(self.model, anchor_vecs, self.nkerns, self.filter_size)
