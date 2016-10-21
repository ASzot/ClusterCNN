import numpy as np

def angle_between(v1, v2):
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

# A helper function to get the anchor vectors of a layer.
def get_anchor_vectors(model0):
    anchor_vectors = []

    for layer in model0.model.layers:
        params = layer.get_weights()
        if len(params) > 0:
            weights = params[0]
            if len(weights.shape) > 2:
                # This is a convolution layer
                add_anchor_vectors = []
                for conv_filter in weights:
                    conv_filter = conv_filter.flatten()
                    add_anchor_vectors.append(conv_filter)
                anchor_vectors.append(add_anchor_vectors)
            else:
                sp = weights.shape
                weights = weights.reshape(sp[1], sp[0])
                anchor_vectors.append(weights)

    return anchor_vectors

def set_anchor_vectors(model, anchor_vectors, nkerns, filter_size):
    # Conolutional layer 0.
    sp = anchor_vectors[0].shape
    anchor_vectors[0] = anchor_vectors[0].reshape(sp[0], nkerns[0], filter_size[0], filter_size[1])

    # Convolutional layer 1.
    sp = anchor_vectors[1].shape
    anchor_vectors[1] = anchor_vectors[1].reshape(sp[0], nkerns[1], filter_size[0], filter_size[1])

    # No need to reshape any of the fully connected layers. They have no associated convolution operation.

    anchor_vectors_index = 0
    for i, layer in enumerate(model.layers):
        params = layer.get_weights()
        if len(params) > 0:
            # This is a layer that has network parameters.
            set_anchor_vector = anchor_vectors[anchor_vectors_index]
            weights = params[0]
            bias = params[1]
            assert set_anchor_vector.shape == weights.shape
            # Does not matter if it is a convolution or fully connected layer.
            model.layers[i].set_weights([set_anchor_vector, bias])
