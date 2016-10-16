import numpy as np

def angle_between(v1, v2):
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
