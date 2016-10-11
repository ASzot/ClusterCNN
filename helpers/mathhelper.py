import numpy as np

def unit_vector(vec):
    return vec / np.linalg.norm(vec)

def angle_between(v1, v2):
    uV1 = unit_vector(v1)
    uV2 = unit_vector(v2)

    return np.arccos(np.clip(np.dot(uV1, uV2), -1.0, 1.0))
