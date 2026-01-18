import numpy as np

def predict(x_samples: np.ndarray, weights: np.ndarray, weights_classes: np.ndarray):
    ps = x_samples @ weights.T
    class_idx = np.argmax(ps, axis=1)
    return weights_classes[class_idx]